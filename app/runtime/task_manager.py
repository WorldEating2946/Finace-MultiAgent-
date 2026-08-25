"""Runtime 任务管理器（PR41 + PR42a Worker Recovery）。

PR41 职责：
    1. submit()  创建任务（INSERT research_tasks → QUEUED）→ 立即返回 research_id
    2. 后台 worker（asyncio task）拾取 → arun_adaptive_research → 状态流转
    3. status / report / resume / cancel 查询与干预
    4. 节点事件 → EventBus（SSE 数据源）

PR42a 新增（Worker Crash Recovery 闭环）：
    - 租约列：worker_id / claimed_at / heartbeat_at / attempts（见 scripts/init_db.py）
    - CAS 认领：queued / stale running / paused → running，原子抢占，仅一个 worker 成功
    - fencing：所有写操作带 WHERE worker_id AND attempts，旧 worker 复活后写被拒
    - heartbeat：worker 并行协程周期续约（lease TTL 内证明存活）
    - startup sweep：进程启动接管孤儿任务（queued / stale running）
    - reaper：watchdog（超时→failed）+ attempts 上限（→failed）+ stale 接管
    - watchdog 用 claimed_at（本代开始时刻），stale 用 heartbeat_at（最近存活）——职责正交

架构：API → ResearchService → TaskManager → ResearchCheckpointer(AsyncPostgresSaver) → Agent

任务状态存 public.research_tasks（业务表），LangGraph 状态存 langgraph.checkpoints。
两者通过 thread_id（= research_id）关联。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
import uuid
from datetime import datetime, timedelta, timezone

import psycopg
import psycopg_pool
from psycopg.rows import dict_row

from app.core.config import settings
from app.core.exceptions import ResearchNotFound
from app.rag.memory import ResearchCheckpointer
from app.runtime.events import EventBus, NodeEvent
from app.runtime.metrics import Metrics
from app.runtime.observability import TaskOutcome, WorkerEvent, log_worker_event
from app.runtime.worker_pool import RedisQueue, WorkerPool

# 状态：task_manager 负责的生命周期（业务表 status）
_QUEUED = "queued"
_RUNNING = "running"
_PAUSED = "paused"
_COMPLETED = "completed"
_REJECTED = "rejected"
_FAILED = "failed"
_CANCELLED = "cancelled"

logger = logging.getLogger(__name__)


def _default_worker_id() -> str:
    """默认 worker 标识：hostname:pid（进程重启 pid 变化 → 旧租约自然失效）。"""
    return f"{socket.gethostname()}:{os.getpid()}"


class TaskManager:
    """异步研究任务编排：提交 / 认领 / 执行 / 心跳 / 恢复 / 取消 / 孤儿回收。"""

    def __init__(
        self,
        checkpointer: ResearchCheckpointer,
        *,
        db_url: str = "",
        event_bus: EventBus | None = None,
        db=None,
        heartbeat_interval: float | None = None,
        lease_ttl: float | None = None,
        reaper_interval: float | None = None,
        max_run_seconds: float | None = None,
        max_attempts: int | None = None,
        worker_id: str | None = None,
        metrics: Metrics | None = None,
        # ── PR43 ──
        redis_client=None,
        queue=None,
        worker_count: int | None = None,
        queue_key: str | None = None,
        shutdown_timeout: float | None = None,
    ) -> None:
        """Args:
        checkpointer: ResearchCheckpointer（AsyncPostgresSaver + 研究图）。
        db_url:      业务库 DSN（research_tasks 表；默认读 config.database_url）。
        event_bus:   事件总线（SSE；默认新建）。
        db:          测试 seam —— FakeTaskDB（None → 生产 psycopg pool）。
        heartbeat_interval / lease_ttl / reaper_interval / max_run_seconds /
        max_attempts / worker_id：测试 seam —— 覆盖 config 默认（None → 读 config）。
        metrics:     PR42b —— 指标注册表（None → 新建；测试注入断言）。
        redis_client: PR43 —— redis.asyncio.Redis 实例（None → 读 config.runtime_redis_url）。
        queue:        PR43 —— 任务队列测试 seam（FakeQueue；None → 生产 RedisQueue）。
        worker_count: PR43 —— 并发 worker 数（None → 读 config.runtime_worker_count）。
        queue_key:    PR43 —— Redis 队列 key（None → 读 config.runtime_redis_queue_key）。
        shutdown_timeout: PR43.5 —— 优雅关闭等待在途任务的超时（None → 读 config）。
        """
        self._cp = checkpointer
        self._db_url = db_url
        self._events = event_bus or EventBus()
        self._db = db
        self._metrics = metrics or Metrics()
        self._active: dict[str, asyncio.Task] = {}  # research_id → worker task
        self._pool: psycopg_pool.AsyncConnectionPool | None = None
        # PR43：Redis 队列 + Worker 池（懒初始化，见 _ensure_worker_pool）
        self._redis_client = redis_client
        self._queue = queue
        self._worker_count = (
            settings.runtime_worker_count if worker_count is None else worker_count
        )
        self._queue_key = (
            settings.runtime_redis_queue_key if queue_key is None else queue_key
        )
        self._shutdown_timeout = (
            settings.runtime_shutdown_timeout
            if shutdown_timeout is None
            else shutdown_timeout
        )
        self._worker_pool: WorkerPool | None = None

        self._worker_id = (
            worker_id or settings.runtime_worker_id or _default_worker_id()
        )
        self._heartbeat_interval = (
            settings.runtime_heartbeat_interval
            if heartbeat_interval is None
            else heartbeat_interval
        )
        self._lease_ttl = settings.runtime_lease_ttl if lease_ttl is None else lease_ttl
        self._reaper_interval = (
            settings.runtime_reaper_interval
            if reaper_interval is None
            else reaper_interval
        )
        self._max_run_seconds = (
            settings.runtime_max_run_seconds
            if max_run_seconds is None
            else max_run_seconds
        )
        self._max_attempts = (
            settings.runtime_max_attempts if max_attempts is None else max_attempts
        )
        # runtime 生命周期标志
        self._started = False
        self._closed = False
        self._reaper_task: asyncio.Task | None = None

    # ── 启动 / 生命周期 ────────────────────────────────────────
    async def ensure_started(self) -> None:
        """幂等启动：建池 → 启动 Worker 池 → 启动扫描（接管孤儿）→ 启动 reaper。

        进程内只执行一次。Worker 池先启动，确保孤儿入队后能被立刻消费。
        """
        if self._started:
            return
        self._started = True
        await self._ensure_pool()
        await self._ensure_worker_pool()
        await self._worker_pool.start()
        await self._sweep_orphans()
        self._reaper_task = asyncio.create_task(self._reaper_loop())

    async def shutdown(self) -> None:
        """取消所有后台 worker + worker 池 + reaper + 关闭连接池（应用下线）。"""
        self._closed = True
        if self._reaper_task is not None:
            self._reaper_task.cancel()
        if self._worker_pool is not None:
            await self._worker_pool.shutdown()
        for task in list(self._active.values()):
            task.cancel()
        if self._active:
            await asyncio.gather(*self._active.values(), return_exceptions=True)
        self._active.clear()
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ── 创建 ────────────────────────────────────────────────────
    async def submit(
        self,
        query: str,
        *,
        company: str = "",
        human_review: bool = False,
    ) -> dict:
        """创建研究任务：INSERT（QUEUED）→ Redis 入队 → 立即返回任务视图。

        PR43：不再由 submit 直接 CAS 认领执行，而是入队交给 Worker 池调度。
        Redis 只负责"这个任务需要执行"，状态真相仍在 PostgreSQL research_tasks。
        Redis 宕机时任务已落 queued，reaper 下一轮扫描补入队（最多一个 reaper 周期）。
        """
        await self.ensure_started()
        research_id = self._new_thread_id(company)
        await self._insert_task(research_id, query, company, human_review)
        self._metrics.inc("research_started_total")  # PR42b 可观测性
        try:
            await self._worker_pool.queue.enqueue(research_id)
        except Exception:  # noqa: BLE001 —— Redis 不可用时降级：reaper 补入队
            logger.warning(
                "event=redis_enqueue_failed research_id=%s worker_id=%s "
                "reason=redis_unreachable 由 reaper 下轮补入队",
                research_id,
                self._worker_id,
            )
        return await self.status(research_id)

    # ── 查询 ────────────────────────────────────────────────────
    async def status(self, research_id: str) -> dict:
        """查询任务视图（业务表为主；checkpoint 状态补充）。"""
        await self.ensure_started()  # 读路径也拉起 recovery（孤儿回收 + reaper）
        row = await self._fetch_task(research_id)
        if row is None:
            raise ResearchNotFound(f"research task not found: {research_id}")
        return self._task_view(row)

    async def get_report(self, research_id: str) -> dict | None:
        """读取研究报告（checkpoint 中 current_report；未生成返回 None）。"""
        await self.ensure_started()
        state = await self._cp.aget_state(research_id)
        if state is None or state.current_report is None:
            return None
        return state.current_report.model_dump()

    async def get_progress(self, research_id: str) -> dict:
        """查询任务 + 进度视图（status / current_step / iteration / missing）。"""
        view = await self.status(research_id)
        state = await self._cp.aget_state(research_id)
        if state is not None:
            view["current_step"] = state.current_step
            view["iteration"] = state.iteration
            view["missing_dimensions"] = list(state.missing_dimensions)
        return view

    # ── 干预 ────────────────────────────────────────────────────
    async def resume(self, research_id: str, *, action: dict) -> dict:
        """人工审核后恢复暂停任务（approve/reject/modify）—— 走统一 CAS 认领。

        与 submit / 崩溃接管同一条 claim 路径：paused → running → Agent Runtime。
        并发两个 resume：CAS 只让一个成功（顺带解决 PR40 的 resume 并发锁缺口）。
        """
        await self.ensure_started()
        row = await self._fetch_task(research_id)
        if row is None:
            raise ResearchNotFound(f"research task not found: {research_id}")
        if row["status"] != _PAUSED:
            # 非暂停任务：直接返回当前状态（幂等）
            return await self.status(research_id)
        claim = await self._claim(research_id)
        if claim is None:
            # 认领失败（并发已被接管）→ 返回当前状态，不重复执行
            return await self.status(research_id)
        task = asyncio.create_task(self._resume_worker(research_id, claim, action))
        self._active[research_id] = task
        return await self.status(research_id)

    async def cancel(self, research_id: str) -> dict:
        """取消排队/运行中任务（已结束/暂停则忽略）。"""
        await self.ensure_started()
        row = await self._fetch_task(research_id)
        if row is None:
            raise ResearchNotFound(f"research task not found: {research_id}")
        if row["status"] not in (_QUEUED, _RUNNING):
            return await self.status(research_id)
        # 运行中：仅当前租约持有者可置 cancelled（fencing）；queued 无主直接置
        self._metrics.inc("research_cancelled_total")  # PR42b
        if row["status"] == _RUNNING and row.get("worker_id") and row.get("attempts"):
            await self._update_fenced(
                research_id,
                row["worker_id"],
                row["attempts"],
                _CANCELLED,
                error=None,
                terminal_reason=TaskOutcome.USER_CANCELLED,
            )
        else:
            await self._update_task(
                research_id,
                status=_CANCELLED,
                terminal_reason=TaskOutcome.USER_CANCELLED,
            )
        log_worker_event(
            WorkerEvent.WORKER_COMPLETED,
            research_id=research_id,
            worker_id=row.get("worker_id") or self._worker_id,
            attempt=row.get("attempts") or 0,
            status=_CANCELLED,
            outcome=TaskOutcome.USER_CANCELLED,
        )
        task = self._active.pop(research_id, None)
        if task is not None:
            task.cancel()
        await self._events.publish(
            NodeEvent.make(research_id, "done", node="", message="任务已取消")
        )
        return await self.status(research_id)

    # ── CAS 认领 ───────────────────────────────────────────────
    async def _claim(self, research_id: str) -> dict | None:
        """原子 CAS 认领：queued / stale running / paused → running，新一代租约。

        只可能有一个 worker 成功（返回新 attempts = 本代租约代次）。
        认领失败返回 None（已被他 worker 抢占 / 租约未过期）。
        """
        pool = await self._ensure_pool()
        now = datetime.now(timezone.utc)
        lease_expiry = now - timedelta(seconds=self._lease_ttl)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE research_tasks
                    SET worker_id=%s, claimed_at=%s, heartbeat_at=%s,
                        status='running', attempts=attempts+1
                    WHERE research_id=%s
                      AND (status='queued'
                           OR (status='running' AND (heartbeat_at IS NULL OR heartbeat_at < %s))
                           OR status='paused')
                    RETURNING research_id, attempts, query, company, human_review, created_at
                    """,
                    (self._worker_id, now, now, research_id, lease_expiry),
                )
                row = await cur.fetchone()
            await conn.commit()
        if row is None:
            return None
        result = dict(row)
        # PR42b 可观测性：认领成功（新任务/接管/resume 统一计数）
        self._metrics.inc("worker_claim_total")
        log_worker_event(
            WorkerEvent.WORKER_CLAIM,
            research_id=research_id,
            worker_id=self._worker_id,
            attempt=result["attempts"],
        )
        # PR43.5 ②：排队等待耗时 = 创建（queued）到 CAS 认领
        self._observe_task_wait(now, result.get("created_at"))
        return result

    def _observe_task_wait(self, claimed_at: datetime, created_at) -> None:
        """记录 task_wait_seconds（创建→认领）。created_at 可为 psycopg datetime
        或 sqlite fake 的 isoformat 字符串。"""
        if created_at is None:
            return
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                return
        if isinstance(created_at, datetime):
            wait = (claimed_at - created_at).total_seconds()
            if wait >= 0:
                self._metrics.observe("task_wait_seconds", wait)

    # ── Worker ──────────────────────────────────────────────────
    async def _run_worker(
        self, research_id: str, claim: dict, *, recovered: bool = False
    ) -> None:
        """后台 worker：CAS 认领后 arun 执行，终态推导 + fencing 写回。

        Args:
            research_id: 任务 id（= thread_id）。
            claim:       _claim 返回的 {attempts, query, company, human_review}。
            recovered:   PR42b —— 本 worker 是否接管 stale 孤儿（结局分类用）。
        """
        attempt = claim["attempts"]
        start = time.monotonic()  # PR42b：research_runtime_seconds 起点

        async def _sink(event: dict):
            """arun_adaptive_research 的 event_sink → EventBus 广播。"""
            await self._events.publish(
                NodeEvent.make(
                    research_id,
                    event.get("type", "progress"),
                    node=event.get("node", ""),
                    message=event.get("message", ""),
                )
            )

        beat = asyncio.create_task(self._heartbeat_loop(research_id, attempt))
        try:
            await self._events.publish(
                NodeEvent.make(
                    research_id, "node_start", node="run", message="研究任务开始"
                )
            )
            await self._cp.arun(
                claim["query"],
                thread_id=research_id,
                human_review=claim["human_review"],
                event_sink=_sink,
            )
            status = await self._derive_terminal(research_id)
            await self._record_terminal(research_id, attempt, status, start, recovered)
            await self._events.publish(
                NodeEvent.make(
                    research_id,
                    "done",
                    node="",
                    message=(
                        "任务完成"
                        if status == _COMPLETED
                        else f"任务{status}（{status}）"
                    ),
                )
            )
        except asyncio.CancelledError:
            raise  # cancel() 主动取消，不标记 FAILED
        except Exception as exc:  # noqa: BLE001 —— 后台 worker 兜底
            await self._record_terminal(
                research_id, attempt, _FAILED, start, recovered, error=str(exc)
            )
            await self._events.publish(
                NodeEvent.make(
                    research_id, "error", node="", message=f"任务失败：{exc}"
                )
            )
        finally:
            beat.cancel()
            self._active.pop(research_id, None)

    async def _record_terminal(
        self, research_id, attempt, status, start, recovered, *, error=None
    ) -> None:
        """终态统一落点（PR42b）：指标 + 结局分类 + 结构化日志 + fencing 写回。"""
        duration = time.monotonic() - start
        outcome = self._outcome_for(status, recovered)
        self._metrics.observe("research_runtime_seconds", duration)
        # PR43.5 ②：Worker 执行耗时（与 research_runtime_seconds 同源，独立命名便于监控）
        self._metrics.observe("worker_execution_seconds", duration)
        if outcome in (TaskOutcome.COMPLETED, TaskOutcome.CRASH_RECOVERED):
            self._metrics.inc("research_completed_total")
        elif outcome == TaskOutcome.FAILED:
            self._metrics.inc("research_failed_total")
        log_worker_event(
            WorkerEvent.WORKER_COMPLETED,
            research_id=research_id,
            worker_id=self._worker_id,
            attempt=attempt,
            status=status,
            outcome=outcome,
            duration=duration,
            error=error,
        )
        await self._finalize(
            research_id,
            attempt,
            status,
            error=error,
            outcome=outcome,
        )

    @staticmethod
    def _outcome_for(status: str, recovered: bool) -> str | None:
        """终态状态 → 结局分类（TaskOutcome）。paused/running 非终态返回 None。"""
        if status == _COMPLETED:
            return TaskOutcome.CRASH_RECOVERED if recovered else TaskOutcome.COMPLETED
        if status == _REJECTED:
            return TaskOutcome.REJECTED
        if status == _FAILED:
            return TaskOutcome.FAILED
        return None

    async def _resume_worker(self, research_id: str, claim: dict, action: dict) -> None:
        """恢复暂停任务：携带人工决策 aresume（同 _run_worker，走 CAS 认领）。"""
        attempt = claim["attempts"]
        start = time.monotonic()  # PR42b

        async def _sink(event: dict):
            await self._events.publish(
                NodeEvent.make(
                    research_id,
                    event.get("type", "progress"),
                    node=event.get("node", ""),
                    message=event.get("message", ""),
                )
            )

        beat = asyncio.create_task(self._heartbeat_loop(research_id, attempt))
        try:
            await self._events.publish(
                NodeEvent.make(
                    research_id,
                    "node_start",
                    node="resume",
                    message="人工审核通过，继续研究",
                )
            )
            await self._cp.aresume(research_id, action=action, event_sink=_sink)
            status = await self._derive_terminal(research_id)
            await self._record_terminal(research_id, attempt, status, start, False)
            await self._events.publish(
                NodeEvent.make(
                    research_id,
                    "done",
                    node="",
                    message="任务完成" if status == _COMPLETED else f"任务{status}",
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._record_terminal(
                research_id, attempt, _FAILED, start, False, error=str(exc)
            )
            await self._events.publish(
                NodeEvent.make(
                    research_id, "error", node="", message=f"任务失败：{exc}"
                )
            )
        finally:
            beat.cancel()
            self._active.pop(research_id, None)

    async def _heartbeat_loop(self, research_id: str, attempt: int) -> None:
        """周期心跳：只有当前租约（worker_id+attempts）能续约；失败不致命（reaper 兜底）。

        已知约束（设计文档 §4c）：与 arun 共享单事件循环。若某节点同步阻塞超过
        lease_ttl，心跳停跳可能导致误判接管 —— lease_ttl 须大于最坏单节点阻塞时长。

        PR42b：租约被接管（rowcount=0）→ WORKER_FENCED + worker_fenced_total + 停跳；
        心跳写异常 → worker_heartbeat_failure_total（静默，reaper 兜底）。
        """
        while not self._closed:
            await asyncio.sleep(self._heartbeat_interval)
            pool = await self._ensure_pool()
            try:
                now = datetime.now(timezone.utc)
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            UPDATE research_tasks SET heartbeat_at=%s
                            WHERE research_id=%s AND worker_id=%s AND attempts=%s
                            """,
                            (now, research_id, self._worker_id, attempt),
                        )
                        updated = cur.rowcount
                    await conn.commit()
                if updated == 0:
                    # 租约已被他 worker 接管（本 worker 已成僵尸）→ 停止心跳
                    self._metrics.inc("worker_fenced_total")
                    log_worker_event(
                        WorkerEvent.WORKER_FENCED,
                        research_id=research_id,
                        worker_id=self._worker_id,
                        attempt=attempt,
                        reason="lease_lost_heartbeat",
                    )
                    break
            except Exception:  # noqa: BLE001 —— 心跳失败静默，reaper 会兜底
                self._metrics.inc("worker_heartbeat_failure_total")
                log_worker_event(
                    WorkerEvent.WORKER_HEARTBEAT,
                    research_id=research_id,
                    worker_id=self._worker_id,
                    attempt=attempt,
                    reason="heartbeat_write_error",
                    level=logging.WARNING,
                )

    async def _derive_terminal(self, research_id: str) -> str:
        """从 checkpoint 推导终态：reject→rejected / pending→paused / end→completed。"""
        state = await self._cp.aget_state(research_id)
        pending = await self._has_pending(research_id)
        if pending:
            return _PAUSED
        if state is None:
            return _RUNNING
        if (state.human_decision or {}).get("action") == "reject":
            return _REJECTED
        if state.next_action == "end":
            return _COMPLETED
        return _RUNNING

    async def _finalize(
        self, research_id: str, attempt: int, status: str, *, error=None, outcome=None
    ) -> None:
        """终态写回（fencing：worker_id+attempts 校验）+ 结局分类落库 + 报告快照原子提交。

        原子性：status 更新与报告快照 INSERT 合并为单事务（先预读 checkpoint 终态）。
        修复：原分两事务写让 status=completed 先于报告可见，负载下偶发"completed 但报告未落"
        竞态（test_submit_runs_to_completed_with_snapshot）；崩溃夹在两事务间更会永久留下
        completed 无报告。现在二者要么一起可见要么一起回滚，SSE done / 状态轮询取报告恒一致。

        若 0 行（租约已被他 worker 接管）→ WORKER_FENCED，放弃写入、不落快照。
        """
        pool = await self._ensure_pool()
        now = datetime.now(timezone.utc)
        # 预读 checkpoint 终态 → 报告快照参数（None = 无报告，不落快照），与 status 同事务
        report_payload = None
        if status in (_COMPLETED, _REJECTED):
            report_payload = await self._build_report_payload(research_id, status)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE research_tasks
                    SET status=%s, error_message=%s, terminal_reason=%s, updated_at=%s
                    WHERE research_id=%s AND worker_id=%s AND attempts=%s
                    """,
                    (
                        status,
                        error,
                        outcome,
                        now,
                        research_id,
                        self._worker_id,
                        attempt,
                    ),
                )
                updated = cur.rowcount
                if updated and report_payload is not None:
                    await cur.execute(
                        """
                        INSERT INTO research_reports
                            (research_id, report, company, query, summary, final_status, completed_at)
                        VALUES (%s, %s::jsonb, %s, %s, %s, %s, %s)
                        ON CONFLICT (research_id) DO UPDATE SET
                            report=EXCLUDED.report, company=EXCLUDED.company,
                            query=EXCLUDED.query, summary=EXCLUDED.summary,
                            final_status=EXCLUDED.final_status, completed_at=EXCLUDED.completed_at
                        """,
                        report_payload,
                    )
            await conn.commit()
        if updated == 0:
            self._metrics.inc("worker_fenced_total")
            log_worker_event(
                WorkerEvent.WORKER_FENCED,
                research_id=research_id,
                worker_id=self._worker_id,
                attempt=attempt,
                reason="lease_lost_finalize",
            )
            return  # 租约已被接管，本 worker 不再写

    async def _build_report_payload(self, research_id: str, status: str):
        """预读 checkpoint 终态 → research_reports 快照 INSERT 参数（upsert）。

        返回 None 表示无报告（agent 未产出），跳过快照写入。
        psycopg3 不原生适配 dict → 用 json 字符串 + %s::jsonb cast；
        FakeTaskDB 会剥掉 ::jsonb 且 json.dumps 兜底，两边行为一致。
        """
        state = await self._cp.aget_state(research_id)
        if state is None or state.current_report is None:
            return None
        report = state.current_report
        return (
            research_id,
            json.dumps(report.model_dump(), ensure_ascii=False, default=str),
            state.target.company,
            state.request,
            (report.summary or "")[:200],
            status,
            datetime.now(timezone.utc),
        )

    # ── Recovery：startup sweep + reaper ─────────────────────────
    async def _sweep_orphans(self) -> None:
        """启动扫描（ensure_started 时执行一次）：接管上一进程遗留的孤儿任务。

        - queued：进程崩于 submit 后、worker 未跑 → CAS 认领调度
        - running 且心跳过期：孤儿 → CAS 接管 → aresume 续跑（checkpoint 幂等）
        - attempts 超限：直接 failed，不再无限重试
        """
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT research_id, status, attempts, worker_id, heartbeat_at "
                    "FROM research_tasks WHERE status IN ('queued','running')",
                )
                rows = [dict(r) for r in await cur.fetchall()]
            await conn.commit()

        for row in rows:
            rid = row["research_id"]
            if row["attempts"] >= self._max_attempts:
                self._metrics.inc("research_failed_total")  # PR42b
                await self._fail_fenced(
                    rid,
                    row.get("worker_id"),
                    row["attempts"],
                    f"max attempts exceeded on recovery ({self._max_attempts})",
                    outcome=TaskOutcome.MAX_ATTEMPTS_EXCEEDED,
                )
                log_worker_event(
                    WorkerEvent.WORKER_STALE,
                    research_id=rid,
                    worker_id=row.get("worker_id") or self._worker_id,
                    attempt=row["attempts"],
                    reason="max_attempts_on_recovery",
                    outcome=TaskOutcome.MAX_ATTEMPTS_EXCEEDED,
                )
                continue
            if row["status"] == _QUEUED:
                # PR43：入队交给 Worker 池调度（不再本地直接认领执行）
                await self._enqueue(rid)
                continue
            # running：本进程刚起无本地 worker —— 心跳过期即孤儿，接管续跑
            age = self._age_seconds(row["heartbeat_at"])
            if age is None or age > self._lease_ttl:
                await self._reclaim(rid, row, age)

    async def _reclaim(
        self, research_id: str, row: dict, heartbeat_age: float | None
    ) -> None:
        """接管 stale 孤儿任务（崩溃恢复）：recovery 指标 + 日志 + 重置入队续跑。

        PR43：不再本地直接认领执行，而是把任务重置为 queued 并重新入队，
        由 Worker 池下一轮捡起执行（CAS 认领 + attempts++ → checkpoint 幂等续跑）。
        recovery_duration = 心跳过期时刻到现在 = 该任务的实际停机时长。
        """
        recovery = heartbeat_age if heartbeat_age is not None else 0.0
        self._metrics.inc("worker_reclaim_total")
        self._metrics.observe("recovery_duration_seconds", recovery)
        log_worker_event(
            WorkerEvent.WORKER_RECLAIM,
            research_id=research_id,
            worker_id=row.get("worker_id") or self._worker_id,
            attempt=row["attempts"],
            reason="heartbeat_expired",
            recovery_duration=recovery,
        )
        await self._reset_to_queued(research_id)
        await self._enqueue(research_id)

    async def _reset_to_queued(self, research_id: str) -> None:
        """把任务重置为 queued 并清空租约列（崩溃恢复的"归还给队列"）。

        不做 fencing：reaper 已判定任务 stale（原 worker 死亡），其复活写会被
        worker_id=NULL + attempts 不变 自然拒绝。
        """
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE research_tasks
                    SET status='queued', worker_id=NULL, claimed_at=NULL,
                        heartbeat_at=NULL, updated_at=%s
                    WHERE research_id=%s
                    """,
                    (datetime.now(timezone.utc), research_id),
                )
            await conn.commit()

    async def _enqueue(self, research_id: str) -> None:
        """入队到 Worker 池队列（Redis 不可用降级：reaper 下轮补入队）。"""
        try:
            await self._worker_pool.queue.enqueue(research_id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "event=redis_enqueue_failed research_id=%s worker_id=%s "
                "reason=redis_unreachable 由 reaper 下轮补入队",
                research_id,
                self._worker_id,
            )

    async def _reaper_loop(self) -> None:
        """reaper 周期任务：watchdog + attempts 上限 + stale 接管（非本地任务）。"""
        while not self._closed:
            await asyncio.sleep(self._reaper_interval)
            try:
                await self._reap()
            except Exception:  # noqa: BLE001 —— reaper 错误不影响主流程
                pass

    async def _sample_queue_length(self) -> None:
        """采样 Redis 队列长度到 Gauge（PR43.5 ②，reaper 周期调用；Redis 故障跳过）。"""
        if self._worker_pool is None:
            return
        try:
            length = await self._worker_pool.queue.length()
            self._metrics.set("queue_length", length)
        except Exception:  # noqa: BLE001 —— Redis 不可用跳过采样，下轮重试
            pass

    async def _reap(self) -> None:
        """单轮巡检：

        - watchdog：claimed_at 超 max_run → failed（含取消本地卡死 worker）
        - attempts 上限：待调度任务已满代次 → failed
        - stale 接管：running 且不在本地 _active、心跳远超过期 → CAS 接管续跑
        """
        await self._sample_queue_length()
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT research_id, status, attempts, worker_id, "
                    "claimed_at, heartbeat_at "
                    "FROM research_tasks WHERE status IN ('queued','running')",
                )
                rows = [dict(r) for r in await cur.fetchall()]
            await conn.commit()

        # 接管阈值：远超单次心跳延迟（多跳未到才判定真死，防同步阻塞误判）
        dead_threshold = self._lease_ttl * 3
        for row in rows:
            rid = row["research_id"]
            if row["status"] == _RUNNING:
                # watchdog：本代次从 claimed_at 起算超时（无论本地或他处 worker）
                claimed_age = self._age_seconds(row["claimed_at"])
                if claimed_age is not None and claimed_age > self._max_run_seconds:
                    task = self._active.pop(rid, None)
                    if task is not None:
                        task.cancel()
                    self._metrics.inc("worker_timeout_total")  # PR42b
                    self._metrics.inc("research_failed_total")
                    log_worker_event(
                        WorkerEvent.WORKER_TIMEOUT,
                        research_id=rid,
                        worker_id=row.get("worker_id") or self._worker_id,
                        attempt=row["attempts"],
                        reason="runtime_exceeded",
                        claimed_age=claimed_age,
                        outcome=TaskOutcome.RUNTIME_TIMEOUT,
                    )
                    await self._fail_fenced(
                        rid,
                        row.get("worker_id"),
                        row["attempts"],
                        f"watchdog timeout (>{self._max_run_seconds}s)",
                        outcome=TaskOutcome.RUNTIME_TIMEOUT,
                    )
                    continue
                # stale 接管：本地拥有的任务跳过（心跳可能因阻塞暂缓）
                if rid in self._active:
                    continue
                hb_age = self._age_seconds(row["heartbeat_at"])
                if hb_age is None or hb_age > dead_threshold:
                    if row["attempts"] >= self._max_attempts:
                        self._metrics.inc("research_failed_total")  # PR42b
                        await self._fail_fenced(
                            rid,
                            row.get("worker_id"),
                            row["attempts"],
                            f"max attempts exceeded ({self._max_attempts})",
                            outcome=TaskOutcome.MAX_ATTEMPTS_EXCEEDED,
                        )
                        log_worker_event(
                            WorkerEvent.WORKER_STALE,
                            research_id=rid,
                            worker_id=row.get("worker_id") or self._worker_id,
                            attempt=row["attempts"],
                            reason="max_attempts",
                            outcome=TaskOutcome.MAX_ATTEMPTS_EXCEEDED,
                        )
                    else:
                        await self._reclaim(rid, row, hb_age)
            elif row["status"] == _QUEUED:
                if row["attempts"] >= self._max_attempts:
                    self._metrics.inc("research_failed_total")  # PR42b
                    await self._fail_fenced(
                        rid,
                        row.get("worker_id"),
                        row["attempts"],
                        f"max attempts exceeded ({self._max_attempts})",
                        outcome=TaskOutcome.MAX_ATTEMPTS_EXCEEDED,
                    )
                    log_worker_event(
                        WorkerEvent.WORKER_STALE,
                        research_id=rid,
                        worker_id=row.get("worker_id") or self._worker_id,
                        attempt=row["attempts"],
                        reason="max_attempts",
                        outcome=TaskOutcome.MAX_ATTEMPTS_EXCEEDED,
                    )
                elif rid not in self._active:
                    await self._enqueue(rid)

    async def _fail_fenced(
        self, research_id, worker_id, attempt, error, *, outcome=None
    ) -> None:
        """带 fencing 的失败写入：worker_id+attempts 校验，防覆盖新租约。

        Args:
            outcome: PR42b —— 结局分类（RUNTIME_TIMEOUT / MAX_ATTEMPTS_EXCEEDED ...），
                     落 research_tasks.terminal_reason。
        """
        pool = await self._ensure_pool()
        now = datetime.now(timezone.utc)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE research_tasks
                    SET status='failed', error_message=%s, terminal_reason=%s, updated_at=%s
                    WHERE research_id=%s AND worker_id=%s AND attempts=%s
                    """,
                    (error, outcome, now, research_id, worker_id, attempt),
                )
            await conn.commit()

    # ── DB 访问（public.research_tasks，连接池）────────────────
    async def _ensure_pool(self):
        """懒初始化业务库连接池（psycopg_pool）；测试注入 FakeTaskDB 时直接返回。"""
        if self._db is not None:
            return self._db
        if self._pool is None:
            self._pool = psycopg_pool.AsyncConnectionPool(
                self._db_url,
                min_size=1,
                max_size=5,
                open=False,
                kwargs={"row_factory": dict_row, "connect_timeout": 10},
            )
            await self._pool.open()
        return self._pool

    # ── Worker 池（PR43：Redis 队列 + 多 Worker）───────────────
    async def _ensure_worker_pool(self) -> None:
        """懒初始化 Worker 池：Redis client → RedisQueue → WorkerPool。

        测试注入 queue=（FakeQueue）时直接用，不连 Redis。
        """
        if self._worker_pool is not None:
            return
        if self._queue is None:
            if self._redis_client is None:
                import redis.asyncio as aioredis

                self._redis_client = aioredis.from_url(
                    settings.runtime_redis_url,
                    decode_responses=False,
                )
            self._queue = RedisQueue(self._redis_client, queue_key=self._queue_key)
        self._worker_pool = WorkerPool(
            self,
            queue=self._queue,
            worker_count=self._worker_count,
            shutdown_timeout=self._shutdown_timeout,
        )

    async def _insert_task(self, research_id, query, company, human_review) -> None:
        pool = await self._ensure_pool()
        now = datetime.now(timezone.utc)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """INSERT INTO research_tasks
                       (research_id, thread_id, status, query, company,
                        human_review, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        research_id,
                        research_id,
                        _QUEUED,
                        query,
                        company,
                        human_review,
                        now,
                        now,
                    ),
                )
            await conn.commit()

    async def _fetch_task(self, research_id) -> dict | None:
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM research_tasks WHERE research_id = %s",
                    (research_id,),
                )
                row = await cur.fetchone()
                return dict(row) if row else None

    async def _update_task(
        self, research_id, *, status, error=None, terminal_reason=None
    ) -> None:
        pool = await self._ensure_pool()
        now = datetime.now(timezone.utc)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """UPDATE research_tasks SET status=%s, error_message=%s,
                       terminal_reason=%s, updated_at=%s WHERE research_id=%s""",
                    (status, error, terminal_reason, now, research_id),
                )
            await conn.commit()

    async def _update_fenced(
        self,
        research_id,
        worker_id,
        attempt,
        status,
        *,
        error=None,
        terminal_reason=None,
    ) -> None:
        """带 fencing 的状态更新（worker_id+attempts 校验），用于取消运行中任务。"""
        pool = await self._ensure_pool()
        now = datetime.now(timezone.utc)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """UPDATE research_tasks SET status=%s, error_message=%s,
                       terminal_reason=%s, updated_at=%s
                       WHERE research_id=%s AND worker_id=%s AND attempts=%s""",
                    (
                        status,
                        error,
                        terminal_reason,
                        now,
                        research_id,
                        worker_id,
                        attempt,
                    ),
                )
            await conn.commit()

    async def _has_pending(self, research_id) -> bool:
        """checkpoint 是否有 pending interrupt（PAUSED 判定）。"""
        try:
            saver = await self._cp._aresolve_cp()
            tuple_ = await saver.aget_tuple(
                config={"configurable": {"thread_id": research_id}}
            )
            return bool(getattr(tuple_, "pending_writes", None))
        except Exception:  # noqa: BLE001
            return False

    # ── 工具 ────────────────────────────────────────────────────
    @staticmethod
    def _age_seconds(ts) -> float | None:
        """时间戳（datetime / iso str）距今秒数；None 输入返回 None。"""
        if ts is None:
            return None
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds()

    def _task_view(self, row: dict) -> dict:
        """DB 行 → API 任务视图（信封 data 字段）。"""
        return {
            "research_id": row["research_id"],
            "thread_id": row["thread_id"],
            "status": row["status"],
            "query": row["query"],
            "company": row["company"],
            "current_step": row.get("current_step") or "",
            "iteration": row.get("iteration") or 0,
            "missing_dimensions": row.get("missing_dimensions") or [],
            "attempts": row.get("attempts") or 0,
            "created_at": self._fmt_ts(row.get("created_at")),
            "updated_at": self._fmt_ts(row.get("updated_at")),
        }

    @staticmethod
    def _fmt_ts(ts) -> str:
        """时间戳格式化：datetime（TIMESTAMPTZ）→ iso；字符串原样透传。"""
        if not ts:
            return ""
        if isinstance(ts, str):
            return ts
        return ts.isoformat()

    @staticmethod
    def _new_thread_id(company: str) -> str:
        """生成 research_id（= thread_id）：r{ts}_{company}_{uuid6}。"""
        comp = company or "research"
        return f"r{int(time.time())}_{comp}_{uuid.uuid4().hex[:6]}"
