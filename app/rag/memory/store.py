"""LangGraph Checkpointer 工厂（PR39）。

存储后端抽象：memory（MemorySaver 进程内） | sqlite（SqliteSaver 本地磁盘）。
生产切换 postgres（PR41）时，仅新增一个 backend 分支，调用方零改动。

msgpack 模块注册：LangGraph 4.1.1 对未注册的 Pydantic 类型反序列化会警告（未来版本
会禁止）。这里通过 JsonPlusSerializer(allowed_msgpack_modules=[...]) 显式注册全部
AgentState 涉及的类型，消除警告并保证向后兼容。
"""

from __future__ import annotations

import sqlite3

from app.core.config import settings

# AgentState checkpoint 涉及的全部 Pydantic 类型（msgpack 显式注册，防未来版本阻断）
ALLOWED_MSGPACK_MODULES: list[tuple[str, str]] = [
    # research schema
    ("app.rag.research.schema", "ResearchTarget"),
    ("app.rag.research.schema", "ResearchIntent"),
    ("app.rag.research.schema", "ResearchDimension"),
    ("app.rag.research.schema", "ResearchStep"),
    ("app.rag.research.schema", "ResearchPlan"),
    # profile schema
    ("app.rag.profile.schema", "EvidenceRef"),
    ("app.rag.profile.schema", "ProfileItem"),
    ("app.rag.profile.schema", "CompanyProfile"),
    # research evaluate / report / state
    ("app.rag.research.evaluate", "ResearchMetrics"),
    ("app.rag.research.evaluate", "ClaimEval"),
    ("app.rag.research.report", "ResearchReport"),
    ("app.rag.research.report", "ReportClaim"),
    ("app.rag.research.state", "Finding"),
    # agent state
    ("app.rag.agent.state", "AgentState"),
]


def _registered_serde():
    """带 msgpack 注册的序列化器（消除 unregistered type 警告）。"""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)


def get_checkpointer(backend: str | None = None, *, db_path: str | None = None):
    """创建 LangGraph checkpointer。

    Args:
        backend: "memory"（默认，进程内）| "sqlite"（本地磁盘）| "postgres"（PR41）。None → 读 config。
        db_path: sqlite 数据库文件路径（backend=sqlite 时用；默认读 config）。

    Returns:
        BaseCheckpointSaver（MemorySaver / SqliteSaver）。

    Note:
        postgres 后端返回的是 **连接池工厂回调**（callable → AsyncPostgresSaver），
        因为 AsyncPostgresSaver 必须在 async 事件循环内构造（__init__ 读 running loop）。
        生产异步入口用 async_get_checkpointer；同步入口（PR40 测试）不触达 postgres 分支。
    """
    backend = backend or settings.rag_checkpoint_store
    if backend == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = db_path or settings.rag_checkpoint_db_path
        # 自己管理连接（长生命周期），不随 with 退出关闭 —— 供跨进程/跨请求复用
        conn = sqlite3.connect(path, check_same_thread=False)
        return SqliteSaver(conn, serde=_registered_serde())
    if backend == "postgres":
        # 返回工厂：async 上下文中调用，返回 AsyncPostgresSaver
        def _postgres_saver_factory() -> "BaseCheckpointSaver":
            return _build_postgres_saver()

        return _postgres_saver_factory
    # 默认 memory
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver(serde=_registered_serde())


async def _build_async_sqlite_saver(db_path: str | None = None):
    """async 上下文构造 AsyncSqliteSaver（PR41 async 方法测试 / 轻量本地路径）。

    SqliteSaver 不支持 async（aget_tuple/ainvoke 会 NotImplementedError），
    PR41 async 方法（arun/aresume）在非 postgres 后端需要 AsyncSqliteSaver。
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    import aiosqlite

    path = db_path or settings.rag_checkpoint_db_path
    conn = await aiosqlite.connect(path)
    return AsyncSqliteSaver(conn, serde=_registered_serde())


async def _build_postgres_saver():
    """在 async 上下文内构造 AsyncPostgresSaver（需 running event loop）。

    连接池复用 psycopg_pool.AsyncConnectionPool，search_path=langgraph 隔离 checkpoint 表。
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    import psycopg_pool

    db_url = settings.checkpoint_db_url
    if not db_url:
        raise ValueError("checkpoint_db_url 未配置（PR41 postgres 后端需要）")
    # checkpoint 表隔离到 langgraph schema（与 public 业务表互不干扰）
    sep = "&" if "?" in db_url else "?"
    pool_url = f"{db_url}{sep}options=-c%20search_path%3Dlanggraph"
    pool = psycopg_pool.AsyncConnectionPool(
        pool_url, min_size=2, max_size=10, open=False
    )
    await pool.open()
    return AsyncPostgresSaver(pool, serde=_registered_serde())
