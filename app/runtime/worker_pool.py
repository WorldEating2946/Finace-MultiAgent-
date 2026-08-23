"""Redis 任务队列 + 多 Worker 池（PR43 + PR43.5 生产加固）。

职责：
    RedisQueue —— Redis list 队列薄封装（RPUSH 入队 / BLPOP 出队）。
    WorkerPool —— N 个 asyncio 逻辑 worker，各自轮询 Redis → CAS 认领 → 执行。

关键设计（PR43 spec）：
    Redis = 调度（"这个任务需要执行"），PostgreSQL = 状态真相（"任务现在什么状态"）。
    Worker 从 Redis 拿到 research_id 后不能直接执行 —— 必须继续 PostgreSQL CAS 认领，
    成功才执行，失败（已被其他 worker 抢占）则丢弃继续轮询。
    这样 PR42a 的 Lease / CAS / Fencing / Heartbeat 全部继续有效。

并发模型：asyncio 逻辑 worker（非线程）。
    每个 worker task 一次只 await 一个 Research（BRPOP → _run_worker），
    所以 worker_count=N → 最多 N 个 Agent 并发 —— 天然限流，不改 _run_worker 内部逻辑。
    该接口设计为可替换（注入 queue= / task_manager=），未来演进 thread/process 池不改上层。

PR43.5 生产加固：
    - 优雅关闭（shutdown_timeout）：停收新任务 → 空闲 worker 立即取消、忙碌 worker
      等在途任务完成（≤timeout）→ 超时强制取消。关闭后已出队任务归还队列。
    - Redis 故障退避：dequeue 异常指数退避重试（1s → max_backoff），Redis 临时不可用
      worker 不死亡，恢复后自动续跑。
    - 指标：worker_active_count（在途执行数 Gauge）+ worker_execution_seconds 分布。
    - 在途任务注册进 TaskManager._active，让 reaper 的"本地拥有跳过"对 PR43 worker 生效
      （防同步阻塞心跳暂缓时被误判接管）。

测试：queue= 注入 FakeQueue（tests/test_worker_pool.py），无需真实 Redis。
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class RedisQueue:
    """Redis list 任务队列：enqueue=RPUSH（尾）/ dequeue=BLPOP（头），近似 FIFO。"""

    def __init__(
        self, redis_client, *, queue_key: str = "finance:research:queue"
    ) -> None:
        """Args:
        redis_client: redis.asyncio.Redis 实例。
        queue_key:    Redis list 键名。
        """
        self._redis = redis_client
        self._key = queue_key

    async def enqueue(self, research_id: str) -> None:
        """入队（append 到队尾）。"""
        await self._redis.rpush(self._key, research_id)

    async def dequeue(self, timeout: float = 5.0) -> str | None:
        """阻塞出队（队头弹出）；超时返回 None。"""
        result = await self._redis.blpop(self._key, timeout=timeout)
        if result is None:
            return None
        return result[1].decode() if isinstance(result[1], bytes) else str(result[1])

    async def length(self) -> int:
        """当前队列长度（测试/监控用）。"""
        return int(await self._redis.llen(self._key))

    async def close(self) -> None:
        """关闭底层连接（WorkerPool 生命周期管理）。"""
        if hasattr(self._redis, "aclose"):
            await self._redis.aclose()
        elif hasattr(self._redis, "close"):
            await self._redis.close()


class WorkerPool:
    """N 个 asyncio 逻辑 worker 池。

    每个 worker 一个永续轮询循环：
        BRPOP → CAS 认领（_claim）→ 成功则 _run_worker（含心跳/终态）→ 回到轮询。
    认领失败（重复出队/已被抢占）→ 静默丢弃继续轮询。
    """

    def __init__(
        self,
        task_manager,
        *,
        queue=None,
        worker_count: int = 4,
        shutdown_timeout: float | None = None,
        max_backoff: float = 30.0,
    ) -> None:
        """Args:
        task_manager:     TaskManager（提供 _claim / _run_worker / _worker_id）。
        queue:            RedisQueue 或 FakeQueue（测试缝；None → 默认 RedisQueue）。
        worker_count:     并发 worker 数。
        shutdown_timeout: 优雅关闭等待在途任务完成的超时（秒；None → 读 config）。
        max_backoff:      Redis 故障重试的退避上限（秒）。
        """
        self._tm = task_manager
        self._queue = queue
        self._worker_count = worker_count
        self._shutdown_timeout = shutdown_timeout
        self._max_backoff = max_backoff
        self._running = False
        self._workers: list[asyncio.Task] = []
        # 正在执行 _run_worker 的 worker index（用于优雅关闭区分忙碌/空闲）
        self._busy: set[int] = set()

    @property
    def queue(self):
        """暴露队列给 TaskManager 调 enqueue()。"""
        return self._queue

    @property
    def worker_count(self) -> int:
        return self._worker_count

    async def start(self) -> None:
        """启动 N 个 worker task（幂等：已启动则直接返回）。"""
        if self._running:
            return
        self._running = True
        for i in range(self._worker_count):
            task = asyncio.create_task(self._worker_loop(i), name=f"worker-pool-{i}")
            self._workers.append(task)

    async def shutdown(self) -> None:
        """优雅关闭（PR43.5 ③）：停收新任务 → 等在途任务完成 → 超时强制取消。

        语义：
            1. _running=False：worker 出队后看到关闭信号会归还任务并退出，不再接新任务。
            2. 空闲 worker（阻塞在 BRPOP/claim 前）立即取消 —— 未持有任务，安全。
            3. 忙碌 worker（正在 _run_worker）等待在途任务自然完成，超过 shutdown_timeout
               强制取消（在途任务留 stale，由 reaper/下次启动接管）。
            4. 最后关闭队列连接。
        """
        if not self._workers:
            self._running = False
            return
        self._running = False
        timeout = 30.0 if self._shutdown_timeout is None else self._shutdown_timeout
        # 先取消空闲 worker（不在 _busy）：它们只是阻塞在等待，没有在途任务
        for i, task in enumerate(self._workers):
            if i not in self._busy:
                task.cancel()
        # 等待忙碌 worker 完成在途任务（≤ shutdown_timeout），超时强制取消
        if self._busy:
            busy_tasks = [t for i, t in enumerate(self._workers) if i in self._busy]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*busy_tasks, return_exceptions=True),
                    timeout=self._shutdown_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "event=shutdown_forced_cancel busy=%d 超过 %ss 强制取消在途任务",
                    len(self._busy),
                    timeout,
                )
                for task in busy_tasks:
                    task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._busy.clear()
        close = getattr(self._queue, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001 —— 关闭失败不阻断下线
                logger.debug("event=queue_close_failed")

    async def _worker_loop(self, index: int) -> None:
        """单个 worker 的永续循环：出队 → CAS 认领 → 执行 → 重复。

        退避（④）：Redis 故障时指数退避重试（1s → max_backoff），成功出队即重置。
        优雅关闭（③）：出队后若已 _running=False，归还任务并退出，不再接新任务。
        指标（②）：在途执行数 worker_active_count + worker_execution_seconds 分布。
        """
        backoff = 1.0
        while self._running:
            try:
                research_id = await self._queue.dequeue(timeout=5.0)
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001 —— Redis 故障不杀 worker，退避重试
                logger.warning(
                    "event=worker_dequeue_error worker=%s:w%d 退避 %ss 后重试",
                    self._tm._worker_id,
                    index,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)
                continue
            backoff = 1.0  # 成功出队 → 重置退避
            if research_id is None:
                continue  # 队列空，等待下轮
            if not self._running:
                # 优雅关闭已开始：归还任务（PG 仍是 queued，下次启动 sweep 兜底）
                try:
                    await self._queue.enqueue(research_id)
                except Exception:  # noqa: BLE001
                    pass
                break
            try:
                claim = await self._tm._claim(research_id)
                if claim is None:
                    continue  # 被其他 worker 抢占（CAS 防重）
                # attempts>1 = 崩溃恢复接管（结局分类 CRASH_RECOVERED）
                recovered = claim["attempts"] > 1
                self._busy.add(index)
                # 注册本地拥有 → reaper 跳过（防同步阻塞心跳暂缓被误判接管）
                self._tm._active[research_id] = asyncio.current_task()
                self._tm._metrics.set("worker_active_count", len(self._busy))
                try:
                    await self._tm._run_worker(research_id, claim, recovered=recovered)
                finally:
                    self._busy.discard(index)
                    self._tm._active.pop(research_id, None)
                    self._tm._metrics.set("worker_active_count", len(self._busy))
            except asyncio.CancelledError:
                self._busy.discard(index)
                self._tm._active.pop(research_id, None)
                self._tm._metrics.set("worker_active_count", len(self._busy))
                break
            except Exception:  # noqa: BLE001 —— 单任务失败不杀 worker
                logger.exception(
                    "event=worker_task_error worker=%s:w%d research_id=%s",
                    self._tm._worker_id,
                    index,
                    research_id,
                )
