"""PR42b 故障测试 + 可观测性断言。

重点覆盖（比普通覆盖率更高价值）：
- 僵尸 worker：租约被接管后，旧 worker 复活的心跳 / complete / failed 三种写全部被拒
- 真并发 reclaim：两个 worker 同时抢 stale 孤儿，仅一个成功（asyncio.gather）
- watchdog timeout 与 crash 区分：心跳正常但运行超 max_run → RUNTIME_TIMEOUT + terminal_reason 落库
- 结局分类（terminal_reason）：COMPLETED / CRASH_RECOVERED / MAX_ATTEMPTS_EXCEEDED / USER_CANCELLED
- 指标计数 + 直方图（research_* / worker_* / runtime / recovery）
- 结构化日志发射（caplog 捕获 worker_claim / worker_completed）
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pytest

from app.runtime.metrics import Metrics
from app.runtime.observability import TaskOutcome
from tests.test_worker_recovery import (
    _fetch,
    _insert_row,
    _mk_tm,
    _wait_status,
    _OLD_TS,
)


async def _expire_heartbeat(db, research_id) -> None:
    """把心跳改过期（模拟 worker 崩溃后无人续约）。"""
    async with db.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE research_tasks SET heartbeat_at=%s WHERE research_id=%s",
                (_OLD_TS, research_id))
        await conn.commit()


# ── 1. 僵尸 worker：三种写全部被拒 ───────────────────────────────

@pytest.mark.asyncio
async def test_zombie_worker_fencing_all_writes_rejected(tmp_path):
    """A 认领 → 心跳过期 → B 接管 → A 复活 → A 的 heartbeat/complete/failed 三种写全被拒。"""
    m_a, m_b = Metrics(), Metrics()
    tmA, db, _ = await _mk_tm(tmp_path, worker_id="wA", metrics=m_a)
    tmB, _, _ = await _mk_tm(tmp_path, worker_id="wB", metrics=m_b)
    rid = "zombie1"
    await _insert_row(db, rid, status="queued", attempts=0)

    claim_a = await tmA._claim(rid)                 # A 认领（attempts=1）
    assert claim_a["attempts"] == 1
    await _expire_heartbeat(db, rid)                # A 崩溃：心跳过期
    claim_b = await tmB._claim(rid)                 # B 接管（attempts=2）
    assert claim_b is not None and claim_b["attempts"] == 2

    # A 复活：心跳写 → 0 行（fencing 拒绝）
    async with db.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE research_tasks SET heartbeat_at=%s "
                "WHERE research_id=%s AND worker_id=%s AND attempts=%s",
                (datetime.now(timezone.utc), rid, "wA", 1))
            hb_cnt = cur.rowcount
        await conn.commit()
    assert hb_cnt == 0

    # A 复活：complete 写（走真实 _finalize 路径）→ 0 行 + worker_fenced_total
    await tmA._finalize(rid, 1, "completed", outcome=TaskOutcome.COMPLETED)
    row = await _fetch(db, rid)
    assert row["status"] == "running" and row["attempts"] == 2   # B 的租约未被覆盖
    assert m_a.worker_fenced_total.value == 1

    # A 复活：failed 写 → 0 行（_fail_fenced 同 fencing）
    await tmA._fail_fenced(rid, "wA", 1, "zombie failed",
                           outcome=TaskOutcome.FAILED)
    row = await _fetch(db, rid)
    assert row["status"] == "running" and row["error_message"] is None

    await tmA.shutdown()
    await tmB.shutdown()


# ── 2. 真并发 reclaim：仅一个成功 ────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_reclaim_only_one_wins(tmp_path):
    """A、B 同时抢同一 stale 孤儿（asyncio.gather 真并发）→ CAS 仅一个成功。"""
    m_a, m_b = Metrics(), Metrics()
    tmA, db, _ = await _mk_tm(tmp_path, worker_id="wA", metrics=m_a)
    tmB, _, _ = await _mk_tm(tmp_path, worker_id="wB", metrics=m_b)
    rid = "race1"
    await _insert_row(db, rid, status="running", attempts=1,
                      worker_id="dead", heartbeat_at=_OLD_TS)

    ca, cb = await asyncio.gather(tmA._claim(rid), tmB._claim(rid))
    winners = sum(1 for c in (ca, cb) if c is not None)
    assert winners == 1                       # 只有一个成功
    winner = ca if ca is not None else cb
    assert winner["attempts"] == 2            # 接管 = 新一代租约
    row = await _fetch(db, rid)
    assert row["attempts"] == 2

    await tmA.shutdown()
    await tmB.shutdown()


# ── 3. watchdog timeout：心跳正常 vs 运行超时 ────────────────────

@pytest.mark.asyncio
async def test_watchdog_timeout_healthy_heartbeat_outcome(tmp_path):
    """心跳新鲜但 claimed_at 超 max_run → watchdog → RUNTIME_TIMEOUT 落库 + 指标。

    与崩溃恢复（CRASH_RECOVERED）职责正交：worker 没死，只是跑太久。
    """
    m = Metrics()
    tm, db, _ = await _mk_tm(tmp_path, metrics=m, max_run_seconds=0.01)
    now_ts = datetime.now(timezone.utc).isoformat()
    await _insert_row(db, "hung1", status="running", attempts=1,
                      worker_id="w-test", claimed_at=_OLD_TS, heartbeat_at=now_ts)

    await tm._reap()
    row = await _fetch(db, "hung1")
    assert row["status"] == "failed"
    assert row["terminal_reason"] == TaskOutcome.RUNTIME_TIMEOUT
    assert m.worker_timeout_total.value == 1
    assert m.research_failed_total.value == 1
    await tm.shutdown()


# ── 4. 结局分类 + 指标 ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_completed_outcome_and_runtime_metric(tmp_path):
    """submit → 完成：COMPLETED 结局 + research_runtime_seconds 直方图。"""
    m = Metrics()
    tm, db, _ = await _mk_tm(tmp_path, metrics=m)
    view = await tm.submit("分析小米", company="小米")
    row = await _wait_status(db, view["research_id"], {"completed"})
    assert row["terminal_reason"] == TaskOutcome.COMPLETED

    assert m.research_started_total.value == 1
    assert m.worker_claim_total.value == 1
    assert m.research_completed_total.value == 1
    assert m.research_failed_total.value == 0
    rt = m.research_runtime_seconds
    assert rt.count == 1 and rt.sum >= 0
    await tm.shutdown()


@pytest.mark.asyncio
async def test_crash_recovery_outcome_and_metrics(tmp_path):
    """崩溃恢复：接管 stale 孤儿 → CRASH_RECOVERED + worker_reclaim_total + recovery 直方图。"""
    m = Metrics()
    tm, db, _ = await _mk_tm(tmp_path, metrics=m)
    await _insert_row(db, "orphan1", status="running", attempts=1,
                      worker_id="dead-worker")

    await tm.ensure_started()
    row = await _wait_status(db, "orphan1", {"completed"})
    assert row["attempts"] == 2
    assert row["terminal_reason"] == TaskOutcome.CRASH_RECOVERED

    assert m.worker_reclaim_total.value == 1
    assert m.worker_claim_total.value == 1           # 接管也是一次认领
    assert m.research_completed_total.value == 1
    rd = m.recovery_duration_seconds
    assert rd.count == 1 and rd.sum > 0              # 停机时长 = 心跳过期→接管
    await tm.shutdown()


@pytest.mark.asyncio
async def test_max_attempts_outcome_persisted(tmp_path):
    """attempts 超限孤儿 → MAX_ATTEMPTS_EXCEEDED + research_failed_total，不再无限重试。"""
    m = Metrics()
    tm, db, _ = await _mk_tm(tmp_path, metrics=m, max_attempts=3)
    await _insert_row(db, "doomed", status="running", attempts=3,
                      worker_id="dead-worker")

    await tm.ensure_started()
    row = await _wait_status(db, "doomed", {"failed"})
    assert row["terminal_reason"] == TaskOutcome.MAX_ATTEMPTS_EXCEEDED
    assert m.research_failed_total.value == 1
    assert m.worker_claim_total.value == 0            # 未再认领
    await tm.shutdown()


@pytest.mark.asyncio
async def test_cancel_outcome_and_counter(tmp_path):
    """用户取消 → USER_CANCELLED + research_cancelled_total。"""
    m = Metrics()
    tm, db, _ = await _mk_tm(tmp_path, metrics=m)
    view = await tm.submit("分析", company="")
    rid = view["research_id"]
    # 重置为 queued（submit 的 worker 可能已秒完成），再取消
    async with db.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE research_tasks SET status='queued' WHERE research_id=%s", (rid,))
        await conn.commit()

    view = await tm.cancel(rid)
    assert view["status"] == "cancelled"
    row = await _fetch(db, rid)
    assert row["terminal_reason"] == TaskOutcome.USER_CANCELLED
    assert m.research_cancelled_total.value == 1
    await tm.shutdown()


# ── 5. 结构化日志 ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_structured_log_events_emitted(tmp_path, caplog):
    """submit→完成：捕获 worker_claim / worker_completed 结构化日志（key=value 可 grep）。"""
    caplog.set_level(logging.INFO, logger="app.runtime.observability")
    tm, db, _ = await _mk_tm(tmp_path)
    view = await tm.submit("分析小米", company="小米")
    await _wait_status(db, view["research_id"], {"completed"})
    await tm.shutdown()

    msgs = [r.getMessage() for r in caplog.records
            if r.name == "app.runtime.observability"]
    claim_msgs = [m for m in msgs if m.startswith("event=worker_claim")]
    done_msgs = [m for m in msgs if m.startswith("event=worker_completed")]
    assert len(claim_msgs) == 1
    assert f"research_id={view['research_id']}" in claim_msgs[0]
    assert len(done_msgs) == 1
    done = done_msgs[0]
    assert f"research_id={view['research_id']}" in done
    assert "status=completed" in done
    assert f"outcome={TaskOutcome.COMPLETED}" in done
    assert "duration=" in done


# ── 6. Metrics Registry 独立验证 ────────────────────────────────

def test_metrics_registry_export_prometheus():
    """Counter/Histogram snapshot + Prometheus 文本导出格式正确。"""
    m = Metrics()
    m.inc("research_started_total")
    m.inc("research_completed_total")
    m.observe("research_runtime_seconds", 12.5)
    m.observe("research_runtime_seconds", 300.0)

    snap = m.snapshot()
    assert snap["research_started_total"] == 1
    assert snap["research_completed_total"] == 1
    rt = snap["research_runtime_seconds"]
    assert rt["count"] == 2 and rt["sum"] == 312.5

    text = m.export_prometheus()
    assert "# TYPE research_completed_total counter" in text
    assert "research_completed_total 1" in text
    assert 'research_runtime_seconds_bucket{le="60"} 1' in text   # 12.5≤60，300>60
    assert 'research_runtime_seconds_bucket{le="+Inf"} 2' in text
    assert "research_runtime_seconds_count 2" in text
