"""Research Checkpoint 编排（PR39 + PR41 async）。

ResearchCheckpointer 包装 LangGraph checkpointer + 自适应研究图：
    - run(thread_id, request)    → 新任务执行（每节点 checkpoint，可跨进程恢复）
    - resume(thread_id, action)  → 从 checkpoint 恢复继续（幂等 / 人工审核续传）
    - record(thread_id)          → 压缩记录（human view，Memory Boundary 派生）
    - arun/aresume/aget_state    → 异步变体（PR41 AsyncPostgresSaver 需要事件循环）

存储后端由 store.get_checkpointer 抽象（memory | sqlite | postgres）：
    - memory/sqlite → 同步方法可直接用（sync run/resume/get_state/record）
    - postgres      → 必须走异步方法（AsyncPostgresSaver 需 running event loop），
                      sync 方法调用时内部自动包一层事件循环（asyncio.run）。
"""

from __future__ import annotations

import asyncio

from langgraph.types import Command

from app.rag.agent import AgentState, build_graph
from app.rag.memory.schema import ResearchRecord, ResumeAction
from app.rag.memory.serializer import to_record
from app.rag.memory.store import get_checkpointer


def _agent_state_from_result(result) -> AgentState:
    """invoke 返回（AgentState / dict / 含 __interrupt__）→ 干净的 AgentState。"""
    if isinstance(result, AgentState):
        return result
    data = dict(result)
    data.pop("__interrupt__", None)  # interrupt 不是 AgentState 字段，剥离
    return AgentState(**data)


class ResearchCheckpointer:
    """研究任务 checkpointer：持久化执行 / 恢复 / 记录（sync + async 双接口）。"""

    def __init__(
        self,
        *,
        backend: str | None = None,
        db_path: str | None = None,
        checkpointer=None,
        tools=None,
        report_builder=None,
    ) -> None:
        """Args:
            backend:        存储后端（"memory"/"sqlite"/"postgres"；None → 读 config）。
            db_path:        sqlite 数据库路径。
            checkpointer:   测试 seam —— 注入已构建的 checkpointer（共享 saver）。
            tools:          测试 seam —— ResearchTools mock。
            report_builder: 测试 seam —— ReportBuilder mock。
        """
        self._backend = backend or "memory"
        self._db_path = db_path
        cp = checkpointer or get_checkpointer(backend=backend, db_path=db_path)
        # postgres → get_checkpointer 返回 factory（callable，需 async 实例化）
        self._cp = cp
        self._tools = tools
        self._report_builder = report_builder

    # ── 内部：解析 checkpointer（factory → 实例）─────────────────
    def _resolve_cp(self):
        """把可能的 factory（postgres）解析成实例。

        postgres 后端必须走 async 方法（AsyncPostgresSaver 需 running event loop），
        同步上下文解析会报错——sync 路径只支持 memory/sqlite。
        """
        if callable(self._cp):
            raise RuntimeError(
                "postgres checkpointer 必须使用 async 方法（arun/aresume/aget_state），"
                "同步 run/resume/get_state 仅支持 memory/sqlite 后端"
            )
        return self._cp

    async def _aresolve_cp(self):
        """async 上下文解析 checkpointer。

        - postgres factory → await 实例化 AsyncPostgresSaver
        - sqlite 实例（SqliteSaver 不支持 async）→ AsyncSqliteSaver（aiosqlite）
        - memory / 注入 saver → 原样（MemorySaver 原生支持 aget_tuple）
        """
        if callable(self._cp):
            return await self._cp()
        from langgraph.checkpoint.sqlite import SqliteSaver

        if isinstance(self._cp, SqliteSaver):
            from app.rag.memory.store import _build_async_sqlite_saver

            return await _build_async_sqlite_saver(db_path=self._db_path)
        return self._cp

    @property
    def is_postgres(self) -> bool:
        """postgres 后端（factory）需要 async 方法。"""
        return callable(self._cp)

    # ── 主入口（sync，向下兼容 PR39/PR40）────────────────────
    def run(self, request: str, *, thread_id: str, human_review: bool = False) -> AgentState:
        """执行研究任务（新线程）：每节点 checkpoint，可跨进程恢复。"""
        from app.rag.agent import run_adaptive_research

        return run_adaptive_research(
            request,
            thread_id=thread_id,
            checkpointer=self._resolve_cp(),
            human_review=human_review,
            _tools=self._tools,
            _report_builder=self._report_builder,
        )

    def resume(
        self,
        thread_id: str,
        *,
        action: ResumeAction | dict | None = None,
    ) -> AgentState | None:
        """从 checkpoint 恢复并继续执行（同步）。

        使用 sync checkpointer（MemorySaver/SqliteSaver）的 get_tuple + graph.invoke，
        避免 asyncio.run 在已运行 event loop（FastAPI TestClient）内冲突。
        """
        graph = build_graph(
            tools=self._tools,
            report_builder=self._report_builder,
            checkpointer=self._resolve_cp(),
        )
        cp = self._resolve_cp()
        if cp.get_tuple(config={"configurable": {"thread_id": thread_id}}) is None:
            return None
        resume_value = self._resume_value(action)
        input_val = Command(resume=resume_value) if resume_value is not None else None
        result = graph.invoke(input_val, config={"thread_id": thread_id})
        return _agent_state_from_result(result)

    def get_state(self, thread_id: str) -> AgentState | None:
        """读取最新 checkpoint 状态（不触发图执行，同步）。"""
        state, _ = self._read_checkpoint(thread_id)
        return state

    def record(self, thread_id: str) -> ResearchRecord | None:
        """获取任务的压缩记录（human view / Memory Boundary 派生）。"""
        state, pending = self._read_checkpoint(thread_id)
        if state is None:
            return None
        return to_record(state, thread_id, pending_writes=pending)

    # ── 主入口（async，PR41 AsyncPostgresSaver）────────────────
    async def arun(
        self,
        request: str,
        *,
        thread_id: str,
        human_review: bool = False,
        event_sink=None,
    ) -> AgentState:
        """异步执行研究任务（生产路径：AsyncPostgresSaver + graph.ainvoke）。

        Args:
            request:     自然语言研究请求。
            thread_id:   任务唯一标识。
            human_review: 证据不足需补步前暂停，等待人工审核。
            event_sink:  异步事件回调（PR41.3 SSE；None 时跳过）。
        """
        from app.rag.agent import arun_adaptive_research

        cp = await self._aresolve_cp()
        return await arun_adaptive_research(
            request,
            thread_id=thread_id,
            checkpointer=cp,
            human_review=human_review,
            _tools=self._tools,
            _report_builder=self._report_builder,
            event_sink=event_sink,
        )

    async def aresume(
        self,
        thread_id: str,
        *,
        action: ResumeAction | dict | None = None,
        event_sink=None,
    ) -> AgentState | None:
        """异步恢复并继续执行。

        - action=None → 纯恢复：中断任务继续；已完成任务幂等返回终态
        - action=dict（HumanDecision 负载）→ 直接作为 resume value（PR40 review_node 人工决策）
        - event_sink：异步事件回调（PR41.3 SSE；None 时跳过）
        """
        from app.rag.agent import build_graph

        cp = await self._aresolve_cp()
        graph = build_graph(
            tools=self._tools,
            report_builder=self._report_builder,
            checkpointer=cp,
        )
        tuple_ = await cp.aget_tuple(config={"configurable": {"thread_id": thread_id}})
        if tuple_ is None:
            return None
        resume_value = self._resume_value(action)
        input_val = Command(resume=resume_value) if resume_value is not None else None
        result = await graph.ainvoke(input_val, config={"thread_id": thread_id})
        return _agent_state_from_result(result)

    async def aget_state(self, thread_id: str) -> AgentState | None:
        """异步读取最新 checkpoint 状态（不触发图执行）。"""
        cp = await self._aresolve_cp()
        tuple_ = await cp.aget_tuple(config={"configurable": {"thread_id": thread_id}})
        if tuple_ is None:
            return None
        channel_values = tuple_.checkpoint.get("channel_values", {})
        return AgentState(**channel_values) if isinstance(channel_values, dict) else channel_values

    async def arecord(self, thread_id: str) -> ResearchRecord | None:
        """异步获取任务的压缩记录。"""
        state = await self.aget_state(thread_id)
        if state is None:
            return None
        cp = await self._aresolve_cp()
        tuple_ = await cp.aget_tuple(config={"configurable": {"thread_id": thread_id}})
        pending = bool(getattr(tuple_, "pending_writes", None))
        return to_record(state, thread_id, pending_writes=pending)

    @staticmethod
    def _resume_value(action: ResumeAction | dict | None):
        """归一化 resume 值：ResumeAction → dict；dict → 原样；None → None。"""
        if action is None:
            return None
        if isinstance(action, ResumeAction):
            return action.model_dump()
        if isinstance(action, dict):
            return action
        raise TypeError(f"action 类型不支持: {type(action).__name__}")

    def _read_checkpoint(self, thread_id: str) -> tuple[AgentState | None, bool]:
        """读 checkpoint：返回 (AgentState, has_pending_interrupt)。"""
        cp = self._resolve_cp()
        tuple_ = cp.get_tuple(config={"configurable": {"thread_id": thread_id}})
        if tuple_ is None:
            return None, False
        channel_values = tuple_.checkpoint.get("channel_values", {})
        state = AgentState(**channel_values) if isinstance(channel_values, dict) else channel_values
        return state, bool(getattr(tuple_, "pending_writes", None))

    def close(self) -> None:
        """关闭底层存储连接（SqliteSaver 持有 sqlite connection，测试/下线时释放）。"""
        # postgres factory：async 实例化的 AsyncPostgresSaver 持有 pool —— 需要 async 关闭
        if callable(self._cp):
            return  # 交由 aclose() 处理（postgres pool 只能在 async 上下文关闭）
        conn = getattr(self._cp, "conn", None)
        if conn is not None:
            conn.close()

    async def aclose(self) -> None:
        """异步关闭（postgres pool / aiosqlite 连接，生产下线用）。"""
        cp = await self._aresolve_cp()
        conn = getattr(cp, "conn", None)
        if conn is None:
            return
        # AsyncPostgresSaver.conn 是 AsyncConnectionPool
        if hasattr(conn, "close") and not hasattr(conn, "execute"):
            await conn.close()
        elif hasattr(conn, "close"):
            await conn.close()
