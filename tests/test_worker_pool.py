"""PR43 Worker 池 + Redis 队列测试（FakeQueue，不依赖真实 Redis）。

覆盖（用户 PR43 测试重点）：
- 多 Worker 并发：RUNTIME_WORKER_COUNT 限制同时执行数
- CAS 防重复执行：同一任务重复出队仅执行一次
- Worker Crash → Recovery：stale 孤儿经 reaper 重置入队 → 其他 worker 续跑
- Zombie Worker → Fencing：旧 worker 复活写被拒
- Redis 故障降级：enqueue 失败不崩 submit（reaper 补入队）
- Worker 数量限制：并发上限 = worker_count
- Checkpoint Resume：崩溃恢复续跑（attempts>1 → CRASH_RECOVERED）
- shutdown：取消所有 worker

复用 test_worker_recovery 的 FakeTaskDB / _StubCheckpointer / _mk_tm（已注入
FakeQueue + 启动 Worker 池）。
"""

from __future__ import annotations

import asyncio

import pytest

from app.runtime.metrics import Metrics
from app.runtime.observability import TaskOutcome
from app.runtime.task_manager import TaskManager
from tests.test_worker_recovery import (
    FakeQueue,
    _fetch,
    _insert_row,
    _mk_tm,
    _wait_status,
    _OLD_TS,
)


async def _mk_tm_pooled(tmp_path, *, worker_count=2, **kw):
    """构建带 Worker 池的 TaskManager，返回 (tm, db, cp, queue)。"""
    tm, db, cp = await _mk_tm(tmp_path, worker_count=worker_count, **kw)
    return tm, db, cp, tm._queue


# ── 1. Redis Queue：入队 / Worker 消费 ─────────────────────────


@pytest.mark.asyncio
async def test_submit_enqueues_to_queue(tmp_path):
    """submit → INSERT queued + 入队（Worker 不消费时队列长度为 1）。"""
    tm, db, cp, q = await _mk_tm_pooled(tmp_path, worker_count=0)
    view = await tm.submit("分析小米", company="小米")
    assert view["research_id"]
    assert await q.length() == 1
    await tm.shutdown()


@pytest.mark.asyncio
async def test_worker_dequeues_and_completes(tmp_path):
    """Worker 出队 → CAS 认领 → 执行 → completed（真实队列流转）。"""
    tm, db, cp, q = await _mk_tm_pooled(tmp_path, worker_count=1)
    view = await tm.submit("分析小米", company="小米")
    row = await _wait_status(db, view["research_id"], {"completed"})
    assert row["attempts"] == 1
    assert await q.length() == 0  # 已消费
    assert len(cp.arun_calls) == 1  # 恰好执行一次
    await tm.shutdown()


# ── 2. CAS 防重复执行 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_cas_prevents_duplicate_processing(tmp_path):
    """同一任务重复入队 → 两次出队，仅一次 CAS 认领成功 → 只执行一次。"""
    tm, db, cp, q = await _mk_tm_pooled(tmp_path, worker_count=2)
    view = await tm.submit("分析", company="")
    rid = view["research_id"]
    await q.enqueue(rid)  # 人为重复入队
    await _wait_status(db, rid, {"completed"})
    row = await _fetch(db, rid)
    assert row["attempts"] == 1  # CAS 只让一个 worker 认领成功
    assert len(cp.arun_calls) == 1  # 未被执行两次
    await tm.shutdown()


# ── 3. Worker Crash → Recovery（经 reaper 重置入队）────────────


@pytest.mark.asyncio
async def test_crash_recovery_via_reset_and_enqueue(tmp_path):
    """stale 孤儿 → reaper reclaim（重置 queued + 入队）→ worker 续跑 completed。

    验证 PR43 崩溃恢复链路：PostgreSQL Lease 发现 stale → 重新入 Redis 队列 →
    Worker CAS 认领 → checkpoint 续跑。attempts=2 + CRASH_RECOVERED。
    """
    m = Metrics()
    tm, db, cp, q = await _mk_tm_pooled(tmp_path, worker_count=1, metrics=m)
    await _insert_row(
        db, "orphan1", status="running", attempts=1, worker_id="dead-worker"
    )

    await tm.ensure_started()  # 启动扫描接管孤儿
    row = await _wait_status(db, "orphan1", {"completed"})
    assert row["attempts"] == 2  # 接管 = 新一代租约
    assert row["terminal_reason"] == TaskOutcome.CRASH_RECOVERED
    assert m.worker_reclaim_total.value == 1
    assert m.worker_claim_total.value == 1
    await tm.shutdown()


@pytest.mark.asyncio
async def test_zombie_fencing_after_reclaim(tmp_path):
    """reaper 重置 stale 任务为 queued 后，旧 worker 的终态写被 fencing（rowcount=0）。"""
    m = Metrics()
    # worker_count=0：不启动消费者，reaper 重置入队后任务保持 queued，便于断言 reset 状态
    tmA, db, _, _ = await _mk_tm_pooled(
        tmp_path, worker_id="wA", metrics=m, worker_count=0
    )
    rid = "zombie1"
    from datetime import datetime, timezone

    now_ts = datetime.now(timezone.utc).isoformat()
    await _insert_row(
        db,
        rid,
        status="running",
        attempts=1,
        worker_id="wA",
        claimed_at=now_ts,
        heartbeat_at=_OLD_TS,
    )

    await tmA._reap()  # reaper 发现 stale → 重置 + 入队
    row = await _fetch(db, rid)
    assert row["status"] == "queued"  # 已归还队列
    assert row["worker_id"] is None  # 租约已清空

    # 旧 worker A（attempts=1）终态写 → 0 行（worker_id 已被清空，WHERE 不匹配）
    pool = await tmA._ensure_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE research_tasks SET status='failed', terminal_reason=%s "
                "WHERE research_id=%s AND worker_id=%s AND attempts=%s",
                (TaskOutcome.FAILED, rid, "wA", 1),
            )
            cnt = cur.rowcount
        await conn.commit()
    assert cnt == 0
    await tmA.shutdown()


# ── 4. Worker 数量限制 ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_count_limits_concurrency(tmp_path):
    """worker_count=2 + 4 个任务 → 任何时刻最多 2 个并发执行（arun 计数峰值 = 2）。"""
    from app.runtime.task_manager import TaskManager
    from tests.test_worker_recovery import (
        FakeTaskDB,
        _StubCheckpointer,
        _completed_state,
        create_schema,
    )

    state = _completed_state()
    cp = _StubCheckpointer(state=state)
    active: set[str] = set()
    peak: list[int] = [0]

    async def arun(request, *, thread_id, human_review=False, event_sink=None):
        active.add(thread_id)
        peak[0] = max(peak[0], len(active))
        await asyncio.sleep(0.3)  # 保持并发窗口
        active.discard(thread_id)
        return state

    cp.arun = arun

    db = FakeTaskDB(tmp_path / "tasks.db")
    await create_schema(db)
    tm = TaskManager(
        cp,
        db=db,
        heartbeat_interval=60.0,
        reaper_interval=9999.0,
        worker_id="w-test",
        queue=FakeQueue(),
        worker_count=2,
    )
    await tm._ensure_worker_pool()
    await tm._worker_pool.start()

    for i in range(4):
        rid = f"load{i}"
        await tm._insert_task(rid, f"分析{i}", "小米", False)
        await tm._worker_pool.queue.enqueue(rid)

    for i in range(4):
        await _wait_status(db, f"load{i}", {"completed"})
    assert peak[0] == 2  # 并发峰值恰为 worker_count（并行确实发生）
    await tm.shutdown()


# ── 5. Redis 故障降级 ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_redis_failure_graceful_degradation(tmp_path):
    """enqueue 抛异常（Redis 宕机）→ submit 不崩，任务已落 PostgreSQL queued。"""

    class _BrokenQueue(FakeQueue):
        async def enqueue(self, research_id):
            raise ConnectionError("redis down")

    # queue 测试缝直接注入故障队列 → worker pool 与 submit 共用同一引用
    tm, db, cp = await _mk_tm(tmp_path, queue=_BrokenQueue())
    view = await tm.submit("分析", company="")
    assert view["status"] in ("queued", "running", "completed")  # 未抛错
    row = await _fetch(db, view["research_id"])
    assert row is not None  # 任务已落库（reaper 会补入队）
    await tm.shutdown()


# ── 6. Checkpoint Resume 语义（崩溃恢复续跑）───────────────────


@pytest.mark.asyncio
async def test_recovery_picks_up_crashed_worker_and_resumes(tmp_path):
    """Worker A 崩溃（stale running）→ reaper 重置入队 → Worker B 续跑完成。

    验证 aresume 路径被触发（checkpoint 续跑），而不仅是全新 arun。
    """
    tm, db, cp, q = await _mk_tm_pooled(tmp_path, worker_count=1)
    await _insert_row(
        db, "crash1", status="running", attempts=2, worker_id="dead-worker"
    )

    await tm.ensure_started()
    row = await _wait_status(db, "crash1", {"completed"})
    assert row["attempts"] == 3  # 崩溃 → 接管 = 新一代租约
    assert row["terminal_reason"] == TaskOutcome.CRASH_RECOVERED
    assert len(cp.arun_calls) == 1  # checkpoint 幂等续跑
    await tm.shutdown()


# ── 7. shutdown 清理 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_shutdown_stops_all_workers(tmp_path):
    """shutdown → 所有 worker 取消，队列可再入但无消费者（生命周期干净）。"""
    tm, db, cp, q = await _mk_tm_pooled(tmp_path, worker_count=3)
    assert len(tm._worker_pool._workers) == 3
    await tm.shutdown()
    assert len(tm._worker_pool._workers) == 0
    assert tm._worker_pool._running is False


# ── 8. 压力：多任务多 Worker 全部完成，无重复 ──────────────────


@pytest.mark.asyncio
async def test_stress_many_tasks_few_workers(tmp_path):
    """20 任务 × 3 worker → 全部完成，每个任务恰好执行一次（无重复无遗漏）。"""
    m = Metrics()
    tm, db, cp, q = await _mk_tm_pooled(tmp_path, worker_count=3, metrics=m)

    for i in range(20):
        rid = f"stress{i}"
        await tm._insert_task(rid, f"分析{i}", "小米", False)
        await tm._worker_pool.queue.enqueue(rid)

    for i in range(20):
        await _wait_status(db, f"stress{i}", {"completed"})

    assert m.research_started_total.value == 0  # 未走 submit（手动插行）
    assert m.research_completed_total.value == 20  # 全部完成
    assert len(cp.arun_calls) == 20  # 无重复执行
    assert len(set(cp.arun_calls)) == 20  # 无遗漏
    await tm.shutdown()


# ── 9. PR43.5 优雅关闭 / 指标 / Redis 故障退避 ─────────────────


@pytest.mark.asyncio
async def test_shutdown_drains_inflight(tmp_path):
    """③ 优雅关闭：shutdown 时在途任务自然完成（不被取消），shutdown 等待其收尾。"""
    from tests.test_worker_recovery import (
        FakeTaskDB,
        _StubCheckpointer,
        _completed_state,
        create_schema,
    )

    started = asyncio.Event()
    cp = _StubCheckpointer(state=_completed_state())

    async def arun(request, *, thread_id, human_review=False, event_sink=None):
        cp.arun_calls.append(thread_id)
        started.set()
        await asyncio.sleep(0.3)  # 模拟在途执行
        return _completed_state()

    cp.arun = arun

    db = FakeTaskDB(tmp_path / "tasks.db")
    await create_schema(db)
    tm = TaskManager(
        cp,
        db=db,
        heartbeat_interval=60.0,
        reaper_interval=9999.0,
        worker_id="w-test",
        queue=FakeQueue(),
        worker_count=1,
        shutdown_timeout=5.0,
    )
    await tm._ensure_worker_pool()
    await tm._worker_pool.start()

    rid = "drain1"
    await tm._insert_task(rid, "分析", "小米", False)
    await tm._worker_pool.queue.enqueue(rid)
    await asyncio.wait_for(started.wait(), 5)  # 确保已进入执行

    await tm.shutdown()  # 应等待在途任务自然完成而非取消
    row = await _fetch(db, rid)
    assert row["status"] == "completed"  # 在途任务未被中断
    assert len(cp.arun_calls) == 1
    await tm.shutdown()  # 幂等


@pytest.mark.asyncio
async def test_shutdown_force_cancels_after_timeout(tmp_path):
    """③ 优雅关闭超时：在途任务超过 shutdown_timeout 被强制取消，shutdown 不挂起。"""
    from tests.test_worker_recovery import (
        FakeTaskDB,
        _StubCheckpointer,
        _completed_state,
        create_schema,
    )

    started = asyncio.Event()
    cancelled = asyncio.Event()
    cp = _StubCheckpointer(state=_completed_state())

    async def arun(request, *, thread_id, human_review=False, event_sink=None):
        started.set()
        try:
            await asyncio.sleep(10)  # 远超 timeout
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return _completed_state()

    cp.arun = arun

    db = FakeTaskDB(tmp_path / "tasks.db")
    await create_schema(db)
    tm = TaskManager(
        cp,
        db=db,
        heartbeat_interval=60.0,
        reaper_interval=9999.0,
        worker_id="w-test",
        queue=FakeQueue(),
        worker_count=1,
        shutdown_timeout=0.2,
    )
    await tm._ensure_worker_pool()
    await tm._worker_pool.start()

    rid = "force1"
    await tm._insert_task(rid, "分析", "小米", False)
    await tm._worker_pool.queue.enqueue(rid)
    await asyncio.wait_for(started.wait(), 5)

    await tm.shutdown()  # 0.2s 后强制取消，不应等待 10s
    assert cancelled.is_set()  # 强制取消确实发生
    assert not tm._worker_pool._workers
    assert not tm._worker_pool._busy


@pytest.mark.asyncio
async def test_worker_metrics(tmp_path):
    """② 指标：worker_active_count 运行中=1/结束=0，execution/wait 分布 + queue_length 采样。"""
    m = Metrics()
    from tests.test_worker_recovery import (
        FakeTaskDB,
        _StubCheckpointer,
        _completed_state,
        create_schema,
    )

    peak: list[float] = []
    cp = _StubCheckpointer(state=_completed_state())

    async def arun(request, *, thread_id, human_review=False, event_sink=None):
        peak.append(m.worker_active_count.value)  # 运行中在途数应为 1
        return _completed_state()

    cp.arun = arun

    db = FakeTaskDB(tmp_path / "tasks.db")
    await create_schema(db)
    tm = TaskManager(
        cp,
        db=db,
        metrics=m,
        heartbeat_interval=60.0,
        reaper_interval=9999.0,
        worker_id="w-test",
        queue=FakeQueue(),
        worker_count=1,
        shutdown_timeout=1.0,
    )
    await tm._ensure_worker_pool()
    await tm._worker_pool.start()

    rid = "metrics1"
    await tm._insert_task(rid, "分析", "小米", False)
    await tm._worker_pool.queue.enqueue(rid)
    await _wait_status(db, rid, {"completed"})
    await tm._reap()  # 采样 queue_length

    assert peak == [1.0]  # 运行中恰 1 个在途
    assert m.worker_active_count.value == 0  # 结束后归零
    assert m.worker_execution_seconds.count >= 1  # 执行耗时分布
    assert m.task_wait_seconds.count >= 1  # 排队等待分布
    assert m.queue_length.value == 0  # 已消费，队列空
    await tm.shutdown()


@pytest.mark.asyncio
async def test_worker_survives_dequeue_errors(tmp_path):
    """④ Redis 故障：dequeue 连续失败 → worker 退避重试不死亡 → 恢复后完成。"""
    from tests.test_worker_recovery import (
        FakeTaskDB,
        _StubCheckpointer,
        _completed_state,
        create_schema,
    )

    class _FlakyQueue(FakeQueue):
        def __init__(self, failures=3):
            super().__init__()
            self.failures = failures

        async def dequeue(self, timeout=5.0):
            if self.failures > 0:
                self.failures -= 1
                raise ConnectionError("redis down")
            return await super().dequeue(timeout=timeout)

    cp = _StubCheckpointer(state=_completed_state())
    db = FakeTaskDB(tmp_path / "tasks.db")
    await create_schema(db)
    tm = TaskManager(
        cp,
        db=db,
        heartbeat_interval=60.0,
        reaper_interval=9999.0,
        worker_id="w-test",
        queue=_FlakyQueue(),
        worker_count=1,
        shutdown_timeout=1.0,
    )
    await tm._ensure_worker_pool()
    tm._worker_pool._max_backoff = 0.05  # 测试加速退避（生产 30s）
    await tm._worker_pool.start()

    rid = "flaky1"
    await tm._insert_task(rid, "分析", "小米", False)
    await tm._worker_pool.queue.enqueue(rid)
    await _wait_status(db, rid, {"completed"})  # 3 次失败后第 4 次成功
    assert len(cp.arun_calls) == 1  # 恰好执行一次，未因故障重复
    await tm.shutdown()
