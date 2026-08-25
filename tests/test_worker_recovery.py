"""PR42a Worker Crash Recovery 测试。

覆盖（用 aiosqlite FakeTaskDB 跑真实 SQL，不依赖真实 postgres）：
- CAS 并发认领：两个 claim 抢同一任务仅一个成功
- fencing：旧代次 worker 的写（heartbeat/terminal）被拒
- submit → 认领 → 执行 → completed + research_reports 快照
- startup sweep：接管孤儿（queued / stale running），attempts 上限判 failed
- reaper：watchdog（超时→failed）/ stale 接管 / 本地 _active 跳过
- resume 走统一 CAS（并发仅一个成功）
- reject → RecordStatus.REJECTED（业务表 + 快照 final_status）
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.runtime.task_manager import TaskManager

# 常数（与 TaskManager 内部一致）
_OLD_TS = "2020-01-01T00:00:00+00:00"


# ── FakeTaskDB（aiosqlite 跑真实 SQL，%s→? 翻译 + 去 jsonb cast）────

class _FakeCursor:
    def __init__(self, cur):
        self._cur = cur

    async def execute(self, sql, params=None):
        sql = sql.replace("::jsonb", "").replace("%s", "?")

        def _adapt(p):
            if isinstance(p, (dict, list)):
                return json.dumps(p, ensure_ascii=False)
            if isinstance(p, datetime):
                return p.isoformat()
            return p

        return await self._cur.execute(sql, [_adapt(p) for p in (params or [])])

    @property
    def rowcount(self):
        return self._cur.rowcount

    async def fetchone(self):
        return await self._cur.fetchone()

    async def fetchall(self):
        return await self._cur.fetchall()


class _FakeConn:
    def __init__(self, aio_conn):
        self._conn = aio_conn

    def cursor(self):
        @asynccontextmanager
        async def _cm():
            cur = await self._conn.cursor()
            try:
                yield _FakeCursor(cur)
            finally:
                await cur.close()

        return _cm()

    async def commit(self):
        return await self._conn.commit()


class FakeTaskDB:
    """TaskManager 测试缝：每次 connection() 新建 aiosqlite 连接（模拟池）。"""

    def __init__(self, path):
        self._path = str(path)

    def connection(self):
        @asynccontextmanager
        async def _cm():
            import sqlite3

            import aiosqlite

            conn = await aiosqlite.connect(self._path)
            conn.row_factory = sqlite3.Row   # dict(row) / row["col"] 与 psycopg dict_row 一致
            try:
                yield _FakeConn(conn)
            finally:
                await conn.close()

        return _cm()


async def create_schema(db) -> None:
    """建业务表（与 scripts/init_db.py 对齐：research_tasks + research_reports）。"""
    async with db.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS research_tasks (
                    research_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'queued',
                    query TEXT NOT NULL,
                    company TEXT NOT NULL DEFAULT '',
                    human_review INTEGER NOT NULL DEFAULT 0,
                    current_step TEXT DEFAULT '',
                    iteration INTEGER DEFAULT 0,
                    missing_dimensions TEXT DEFAULT '[]',
                    error_message TEXT,
                    worker_id TEXT,
                    claimed_at TEXT,
                    heartbeat_at TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    terminal_reason TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS research_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    research_id TEXT NOT NULL UNIQUE,
                    report TEXT NOT NULL,
                    company TEXT NOT NULL DEFAULT '',
                    query TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    final_status TEXT NOT NULL DEFAULT 'completed',
                    completed_at TEXT,
                    created_at TEXT
                )
            """)
        await conn.commit()


# ── Stub Checkpointer（TaskManager 执行路径桩）──────────────────

class _StubTuple:
    def __init__(self, pending):
        self.pending_writes = pending


class _StubSaver:
    def __init__(self, pending=False):
        self._pending = pending

    async def aget_tuple(self, config=None):
        return _StubTuple(self._pending)


class _StubCheckpointer:
    """TaskManager 测试桩：arun/aresume 秒完成，aget_state 返回可配置状态。"""

    def __init__(self, *, state=None, pending=False, arun_impl=None):
        self.state = state if state is not None else _completed_state()
        self._pending = pending
        self._arun_impl = arun_impl or (lambda rid: None)

        async def _noop(rid):
            return None

        self._arun_impl = _noop if arun_impl is None else arun_impl
        self.arun_calls: list[str] = []
        self.aresume_calls: list[str] = []

    @property
    def is_postgres(self):
        return True

    async def arun(self, request, *, thread_id, human_review=False, event_sink=None):
        self.arun_calls.append(thread_id)
        await self._arun_impl(thread_id)
        return self.state

    async def aresume(self, thread_id, *, action=None, event_sink=None):
        self.aresume_calls.append(thread_id)
        await self._arun_impl(thread_id)
        return self.state

    async def aget_state(self, thread_id):
        return self.state

    async def _aresolve_cp(self):
        return _StubSaver(pending=self._pending)


# ── 状态工厂 ─────────────────────────────────────────────────────

def _completed_state():
    """next_action=end → 终态 COMPLETED，且带 current_report（供快照）。"""
    from app.rag.research.report import ReportClaim, ResearchReport

    report = ResearchReport(
        title="t", summary="研究摘要",
        advantages=[ReportClaim(claim="优势", evidence=[])],
        risks=[], uncertainties=[], evidence=[],
    )
    return SimpleNamespace(
        next_action="end",
        human_decision={},
        current_report=report,
        target=SimpleNamespace(company="小米"),
        request="分析小米汽车",
    )


def _rejected_state():
    s = _completed_state()
    s.human_decision = {"action": "reject"}
    return s


# ── 工具 ─────────────────────────────────────────────────────────

class FakeQueue:
    """Worker 池测试缝：asyncio.Queue 包装，接口与 RedisQueue 一致（PR43）。

    enqueue / dequeue(timeout) / length，无需真实 Redis。
    """

    def __init__(self):
        self._q: asyncio.Queue = asyncio.Queue()

    async def enqueue(self, research_id: str) -> None:
        await self._q.put(research_id)

    async def dequeue(self, timeout: float = 5.0) -> str | None:
        try:
            return await asyncio.wait_for(self._q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def length(self) -> int:
        return self._q.qsize()


async def _mk_tm(tmp_path, *, state=None, pending=False, **kw):
    """构建 TaskManager：FakeTaskDB + stub checkpointer + FakeQueue + 启动 Worker 池。

    PR43：submit 改走 Redis 队列（测试用 FakeQueue 代替）→ 需启动 Worker 池消费。
    返回 (tm, db, cp)；队列可通过 tm._queue 取（FakeQueue）。
    """
    db = FakeTaskDB(tmp_path / "tasks.db")
    await create_schema(db)
    cp = _StubCheckpointer(state=state, pending=pending)
    defaults = dict(
        heartbeat_interval=60.0,   # 测试默认不自动心跳
        reaper_interval=9999.0,    # 手动调 _reap，reaper 循环不干扰
        lease_ttl=30.0,
        max_run_seconds=600.0,
        max_attempts=3,
        worker_id="w-test",
        queue=FakeQueue(),         # PR43：队列测试缝（不连真实 Redis）
        worker_count=1,            # PR43：单 worker，测试行为确定
    )
    defaults.update(kw)
    tm = TaskManager(cp, db=db, **defaults)
    # PR43：立即启动 Worker 池（即使测试直接调 _reap / _claim 也有消费者在轮询）
    await tm._ensure_worker_pool()
    await tm._worker_pool.start()
    return tm, db, cp


async def _wait_status(db, research_id, statuses, timeout=3.0):
    """轮询业务表直到状态命中，返回该行 dict。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        async with db.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT status, attempts, error_message, terminal_reason "
                    "FROM research_tasks WHERE research_id=%s", (research_id,))
                row = await cur.fetchone()
            await conn.commit()
        if row is not None and row["status"] in statuses:
            return dict(row)
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"task {research_id} 未到达 {statuses}; 最后={dict(row) if row else None}")


async def _insert_row(db, research_id, *, status, attempts=0,
                      worker_id=None, claimed_at=_OLD_TS, heartbeat_at=_OLD_TS):
    async with db.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """INSERT INTO research_tasks
                   (research_id, thread_id, status, query, company, human_review,
                    worker_id, claimed_at, heartbeat_at, attempts, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (research_id, research_id, status, "分析", "小米", 0,
                 worker_id, claimed_at, heartbeat_at, attempts, _OLD_TS, _OLD_TS))
        await conn.commit()


async def _fetch(db, research_id) -> dict:
    async with db.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT * FROM research_tasks WHERE research_id=%s", (research_id,))
            row = await cur.fetchone()
        await conn.commit()
    return dict(row)


# ── 测试 ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cas_claim_only_one_wins(tmp_path):
    """CAS 认领：queued 任务两个 worker 并发抢，仅一个成功。"""
    tm, db, cp = await _mk_tm(tmp_path)
    await tm._insert_task("r1", "分析", "小米", False)

    r1 = await tm._claim("r1")
    r2 = await tm._claim("r1")   # 已 running 且心跳新鲜 → 认领失败
    assert r1 is not None and r1["attempts"] == 1
    assert r2 is None
    await tm.shutdown()


@pytest.mark.asyncio
async def test_fencing_old_worker_write_rejected(tmp_path):
    """fencing：Worker A 崩溃 → B 接管（attempts=2）→ A 的旧代次写被拒（0 行）。"""
    tm, db, cp = await _mk_tm(tmp_path)
    await tm._insert_task("r1", "分析", "小米", False)
    claim1 = await tm._claim("r1")
    assert claim1["attempts"] == 1

    # 模拟 A 崩溃：心跳过期 → B 接管
    async with db.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE research_tasks SET heartbeat_at=%s WHERE research_id=%s",
                (_OLD_TS, "r1"))
        await conn.commit()
    claim2 = await tm._claim("r1")
    assert claim2 is not None and claim2["attempts"] == 2

    # 旧 worker A（attempts=1）的终态写 → 0 行（fencing 拒绝）
    pool = await tm._ensure_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE research_tasks SET status='completed' "
                "WHERE research_id=%s AND worker_id=%s AND attempts=%s",
                ("r1", "w-test", 1))
            cnt = cur.rowcount
        await conn.commit()
    assert cnt == 0
    await tm.shutdown()


@pytest.mark.asyncio
async def test_submit_runs_to_completed_with_snapshot(tmp_path):
    """submit → CAS 认领 → worker 执行 → completed + research_reports 快照。"""
    tm, db, cp = await _mk_tm(tmp_path)
    view = await tm.submit("分析小米", company="小米")
    assert view["status"] in ("queued", "running", "completed")  # stub 秒完成
    row = await _wait_status(db, view["research_id"], {"completed"})
    assert row["attempts"] == 1
    assert cp.arun_calls == [view["research_id"]]

    async with db.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT final_status, company, query FROM research_reports "
                "WHERE research_id=%s", (view["research_id"],))
            rep = await cur.fetchone()
        await conn.commit()
    assert rep is not None
    assert rep["final_status"] == "completed"
    assert rep["company"] == "小米"
    await tm.shutdown()


@pytest.mark.asyncio
async def test_startup_sweep_reclaims_stale_running(tmp_path):
    """startup sweep：接管上一进程遗留的 stale running 孤儿 → 续跑 completed。"""
    db = FakeTaskDB(tmp_path / "tasks.db")
    await create_schema(db)
    await _insert_row(db, "orphan1", status="running", attempts=1,
                      worker_id="dead-worker")
    tm, _, cp = await _mk_tm(tmp_path)   # 同一 tmp_path 文件，ensure_started 时扫描

    await tm.ensure_started()
    row = await _wait_status(db, "orphan1", {"completed"})
    assert row["attempts"] == 2          # 接管 = 新一代租约
    await tm.shutdown()


@pytest.mark.asyncio
async def test_startup_sweep_dispatches_queued(tmp_path):
    """startup sweep：queued 任务（进程崩于 submit 后）→ 认领执行。"""
    db = FakeTaskDB(tmp_path / "tasks.db")
    await create_schema(db)
    await _insert_row(db, "queued1", status="queued", attempts=0)
    tm, _, cp = await _mk_tm(tmp_path)

    await tm.ensure_started()
    row = await _wait_status(db, "queued1", {"completed"})
    assert row["attempts"] == 1
    await tm.shutdown()


@pytest.mark.asyncio
async def test_max_attempts_fails_orphan(tmp_path):
    """attempts 已达上限的孤儿 → sweep 判 failed，不再无限重试。"""
    db = FakeTaskDB(tmp_path / "tasks.db")
    await create_schema(db)
    await _insert_row(db, "doomed", status="running", attempts=3,
                      worker_id="dead-worker")
    tm, _, cp = await _mk_tm(tmp_path, max_attempts=3)

    await tm.ensure_started()
    row = await _wait_status(db, "doomed", {"failed"})
    assert "max attempts" in (row["error_message"] or "")
    await tm.shutdown()


@pytest.mark.asyncio
async def test_watchdog_fails_hung_task(tmp_path):
    """reaper watchdog：claimed_at 超 max_run → failed（不因 heartbeat 新鲜而豁免）。"""
    tm, db, cp = await _mk_tm(tmp_path, max_run_seconds=0.01)
    await _insert_row(db, "hung1", status="running", attempts=1, worker_id="w-test")
    # claimed_at 保持很旧（2020），heartbeat 新鲜 —— 只有 claimed_at 能暴露"跑太久"

    await tm._reap()   # 手动触发单轮巡检
    row = await _wait_status(db, "hung1", {"failed"})
    assert "watchdog" in (row["error_message"] or "")
    await tm.shutdown()


@pytest.mark.asyncio
async def test_reap_stale_reclaims_non_local(tmp_path):
    """reaper stale 接管：本代刚起（claimed_at 新鲜）但心跳远超过期 → 认领续跑。

    watchdog（claimed_at）与 stale（heartbeat_at）职责正交：本代未超时、
    但 zombie 无心跳 → 接管而非判死。
    """
    tm, db, cp = await _mk_tm(tmp_path)
    now_ts = datetime.now(timezone.utc).isoformat()
    await _insert_row(db, "stale1", status="running", attempts=1,
                      worker_id="dead-worker",
                      claimed_at=now_ts, heartbeat_at=_OLD_TS)

    await tm._reap()
    row = await _wait_status(db, "stale1", {"completed"})
    assert row["attempts"] == 2
    await tm.shutdown()


@pytest.mark.asyncio
async def test_reap_skips_local_active(tmp_path):
    """reaper stale 接管跳过本地 _active 任务（防同步阻塞心跳暂缓误判）。"""
    tm, db, cp = await _mk_tm(tmp_path)
    now_ts = datetime.now(timezone.utc).isoformat()
    await _insert_row(db, "local1", status="running", attempts=1,
                      worker_id="w-test", claimed_at=now_ts, heartbeat_at=_OLD_TS)
    # 模拟本地拥有：_active 里有该任务
    tm._active["local1"] = asyncio.create_task(asyncio.sleep(9999))

    await tm._reap()
    row = await _fetch(db, "local1")
    assert row["status"] == "running"
    assert row["attempts"] == 1    # 未被接管

    tm._active["local1"].cancel()
    await tm.shutdown()


@pytest.mark.asyncio
async def test_resume_via_cas(tmp_path):
    """paused → resume 走统一 CAS 认领 → 续跑 completed，并发仅一个成功。"""
    tm, db, cp = await _mk_tm(tmp_path)
    await _insert_row(db, "paused1", status="paused", attempts=1,
                      worker_id="old-worker")

    await tm.ensure_started()
    view = await tm.resume("paused1", action={"action": "approve", "feedback": "ok"})
    assert view["status"] in ("running", "completed")
    row = await _wait_status(db, "paused1", {"completed"})
    assert row["attempts"] == 2            # resume = 新一代租约
    assert "paused1" in cp.aresume_calls
    await tm.shutdown()


@pytest.mark.asyncio
async def test_resume_concurrent_only_one_claims(tmp_path):
    """并发两个 resume：CAS 仅让一个认领成功（旧 resume 并发锁缺口）。"""
    tm, db, cp = await _mk_tm(tmp_path)
    await _insert_row(db, "paused2", status="paused", attempts=1,
                      worker_id="old-worker")

    c1 = await tm._claim("paused2")
    c2 = await tm._claim("paused2")   # 已 running → 认领失败
    assert c1 is not None and c1["attempts"] == 2
    assert c2 is None
    await tm.shutdown()


@pytest.mark.asyncio
async def test_reject_derives_rejected_state(tmp_path):
    """reject → 业务表 rejected + 快照 final_status=rejected（不再伪装 completed）。"""
    tm, db, cp = await _mk_tm(tmp_path, state=_rejected_state())
    view = await tm.submit("分析小米", company="小米")
    row = await _wait_status(db, view["research_id"], {"rejected"})
    assert row["status"] == "rejected"

    # 状态先落、快照后落（两个事务），轮询等快照就绪，避免竞态空窗
    rid = view["research_id"]
    deadline = time.monotonic() + 3.0
    rep = None
    while time.monotonic() < deadline:
        async with db.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT final_status FROM research_reports WHERE research_id=%s",
                    (rid,))
                rep = await cur.fetchone()
            await conn.commit()
        if rep is not None:
            break
        await asyncio.sleep(0.02)
    assert rep is not None, "reject 后 research_reports 快照未落库"
    assert rep["final_status"] == "rejected"
    await tm.shutdown()
