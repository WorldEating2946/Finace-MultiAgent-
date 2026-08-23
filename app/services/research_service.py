"""Research Service 层（PR40 sync + PR41 async）。

API → Service → Agent → Memory 的中间层——API 不直接耦合 LangGraph/Agent。

双接口：
    sync  （create_task / get_task / get_report / resume）—— PR40 兼容，走 checkpointer 直调，
            单测/脚本可直接用（返回 completed）。
    async （acreate_task / aget_task / aget_report / aresume / acancel）—— PR41 生产路径，
            走 TaskManager（后台 worker + PostgreSQL 业务表 + SSE 事件总线），
            POST /start 立即返回 queued。

状态映射：RecordStatus（queued/running/paused/completed/failed/cancelled）→ API status
（waiting_human 兼容 PR40 语义：paused 对外仍显示 waiting_human）。
"""

from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, Field

from app.core.exceptions import ResearchNotFound
from app.rag.memory import ResearchCheckpointer
from app.rag.memory.schema import RecordStatus
from app.runtime import TaskManager

# RecordStatus → API 对外 status（PR40 语义保持：paused → waiting_human）
_STATUS_MAP = {
    RecordStatus.QUEUED: "queued",
    RecordStatus.RUNNING: "running",
    RecordStatus.PAUSED: "waiting_human",
    RecordStatus.COMPLETED: "completed",
    RecordStatus.REJECTED: "rejected",
    RecordStatus.FAILED: "failed",
    RecordStatus.CANCELLED: "cancelled",
}


class ResearchTask(BaseModel):
    """研究任务视图（API 对外返回）。"""

    research_id: str
    thread_id: str
    status: str                       # queued | running | waiting_human | completed | failed | cancelled
    query: str
    company: str
    current_step: str = ""
    iteration: int = 0
    missing_dimensions: list[str] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class ResearchService:
    """研究任务编排服务：创建 / 查询 / 报告 / 恢复 / 取消。

    - sync 模式（PR40）：checkpointer 直调，任务阻塞到完成。
    - async 模式（PR41）：TaskManager 后台 worker，立即返回 queued。
    """

    def __init__(
        self,
        *,
        tools=None,
        report_builder=None,
        checkpointer: ResearchCheckpointer | None = None,
        task_manager: TaskManager | None = None,
        db_url: str = "",
    ) -> None:
        """Args:
            tools:          测试 seam —— ResearchTools mock。
            report_builder: 测试 seam —— ReportBuilder mock。
            checkpointer:   ResearchCheckpointer（默认从 config 构建）。
            task_manager:   PR41 —— TaskManager（异步 runtime）；None 时惰性构建。
            db_url:         PR41 —— 业务库 DSN（TaskManager 用）。
        """
        self._tools = tools
        self._report_builder = report_builder
        self._checkpointer = checkpointer or ResearchCheckpointer(
            tools=tools, report_builder=report_builder
        )
        self._task_manager = task_manager
        self._db_url = db_url

    @property
    def event_bus(self):
        """SSE 事件总线（PR41.3 stream 端点订阅）。"""
        return self._tm()._events

    def _tm(self) -> TaskManager:
        """惰性构建 TaskManager（async 模式）。"""
        if self._task_manager is None:
            self._task_manager = TaskManager(
                self._checkpointer, db_url=self._db_url or _default_db_url()
            )
        return self._task_manager

    # ── sync 模式（PR40 兼容：单测 / 脚本直接调用）───────────
    def create_task(
        self,
        query: str,
        *,
        company: str = "",
        human_review: bool = False,
    ) -> ResearchTask:
        """创建并执行研究任务（同步，阻塞到完成或暂停）。"""
        thread_id = self._new_thread_id(company)
        self._checkpointer.run(
            query, thread_id=thread_id, human_review=human_review
        )
        return self._require_task(thread_id)

    def get_task(self, research_id: str) -> ResearchTask | None:
        """查询任务状态（sync，checkpointer record 推导）。"""
        rec = self._checkpointer.record(research_id)
        if rec is None:
            return None
        return ResearchTask(
            research_id=research_id,
            thread_id=research_id,
            status=_STATUS_MAP.get(rec.status, "running"),
            query=rec.query,
            company=rec.company,
            current_step=rec.current_step,
            iteration=rec.iteration,
            missing_dimensions=list(rec.missing_dimensions),
            created_at=rec.created_at,
            updated_at=rec.updated_at,
        )

    def get_report(self, research_id: str) -> dict | None:
        """读取任务报告（sync）。"""
        state = self._checkpointer.get_state(research_id)
        if state is None or state.current_report is None:
            return None
        return state.current_report.model_dump()

    def resume(self, research_id: str, *, action: dict) -> ResearchTask:
        """从 checkpoint 恢复继续执行（sync，携带人工审核决策）。"""
        self._checkpointer.resume(research_id, action=action)
        return self._require_task(research_id)

    # ── async 模式（PR41 生产路径：后台 worker + 立即返回）────
    async def acreate_task(
        self,
        query: str,
        *,
        company: str = "",
        human_review: bool = False,
    ) -> ResearchTask:
        """创建研究任务。

        - postgres 后端（PR41 生产）：TaskManager 后台 worker，立即返回 queued。
        - memory/sqlite 后端（兼容 PR40 测试 / 本地）：同步执行，直接返回终态。
        """
        if not self._checkpointer.is_postgres:
            return self.create_task(query, company=company, human_review=human_review)
        view = await self._tm().submit(
            query, company=company, human_review=human_review
        )
        return self._view_to_task(view)

    async def aget_task(self, research_id: str) -> ResearchTask | None:
        """查询任务状态（异步）。"""
        if not self._checkpointer.is_postgres:
            return self.get_task(research_id)
        try:
            view = await self._tm().get_progress(research_id)
        except ResearchNotFound:
            return None
        return self._view_to_task(view)

    async def aget_report(self, research_id: str) -> dict | None:
        """读取任务报告（异步）。"""
        if not self._checkpointer.is_postgres:
            return self.get_report(research_id)
        return await self._tm().get_report(research_id)

    async def aresume(self, research_id: str, *, action: dict) -> ResearchTask:
        """人工审核后恢复暂停任务（异步）。"""
        if not self._checkpointer.is_postgres:
            return self.resume(research_id, action=action)
        view = await self._tm().resume(research_id, action=action)
        return self._view_to_task(view)

    async def acancel(self, research_id: str) -> ResearchTask:
        """取消排队/运行中任务（异步）。"""
        if not self._checkpointer.is_postgres:
            # sqlite/memory：无后台 worker，直接返回当前状态
            task = self.get_task(research_id)
            if task is None:
                raise ResearchNotFound(f"research task not found: {research_id}")
            return task
        view = await self._tm().cancel(research_id)
        return self._view_to_task(view)

    async def shutdown(self) -> None:
        """关闭 TaskManager（后台 worker + 业务池）+ checkpointer（checkpoint 池）。"""
        if self._task_manager is not None:
            await self._task_manager.shutdown()
        await self._checkpointer.aclose()

    # ── 内部 ───────────────────────────────────────────────────
    @staticmethod
    def _view_to_task(view: dict) -> ResearchTask:
        """TaskManager 任务视图 dict → ResearchTask。"""
        return ResearchTask(
            research_id=view["research_id"],
            thread_id=view["thread_id"],
            status=view["status"],
            query=view.get("query", ""),
            company=view.get("company", ""),
            current_step=view.get("current_step", ""),
            iteration=view.get("iteration", 0),
            missing_dimensions=list(view.get("missing_dimensions") or []),
            created_at=view.get("created_at", ""),
            updated_at=view.get("updated_at", ""),
        )

    @staticmethod
    def _new_thread_id(company: str) -> str:
        """生成 research_id（= thread_id）：r{ts}_{company}_{uuid6}。"""
        comp = company or "research"
        return f"r{int(time.time())}_{comp}_{uuid.uuid4().hex[:6]}"

    def _require_task(self, research_id: str) -> ResearchTask:
        task = self.get_task(research_id)
        if task is None:
            raise ResearchNotFound(f"research task not found: {research_id}")
        return task


def _default_db_url() -> str:
    """业务库 DSN（config.database_url，未配置回退空——TaskManager 报错）。"""
    from app.core.config import settings

    return settings.database_url
