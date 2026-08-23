"""
app/database/session.py — 业务库异步 ORM 会话层（JWT 认证用户表用）

复用项目 PostgreSQL 业务库（settings.database_url / CHECKPOINT_DB_URL 同源），
仅新增 users 表（Base.metadata.create_all），不改动既有 research_tasks 等表。

对外入口：
    get_db            — FastAPI 依赖，yield 一个 AsyncSession
    init_db           — 启动建表（create_all，幂等）
    get_engine        — 惰性单例 async engine

DSN 解析：优先 settings.database_url；为空则用 PG 环境变量回退拼接。
自动规整 `postgresql://` → `postgresql+asyncpg://`（asyncpg 驱动必需）。
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """异步 ORM 声明基类（所有表模型继承）。"""


def _normalize_dsn(dsn: str) -> str:
    """确保使用 asyncpg 异步驱动前缀。

    settings.database_url 可能是 `postgresql://`（sync 形态），
    create_async_engine 必须 `postgresql+asyncpg://`。
    """
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    if dsn.startswith("postgres://"):
        return dsn.replace("postgres://", "postgresql+asyncpg://", 1)
    return dsn


def _resolve_dsn() -> str:
    """业务库 DSN：settings.database_url 优先，空则 PG 环境变量兜底。"""
    if settings.database_url:
        return _normalize_dsn(settings.database_url)
    user = os.getenv("POSTGRES_USER", "finance")
    pwd = os.getenv("POSTGRES_PASSWORD", "CHANGE_ME")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "finance_agent")
    return f"postgresql+asyncpg://{user}:{pwd}@{host}:{port}/{db}"


# ── 惰性单例 ──────────────────────────────────────────────
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """惰性创建 async engine（DSN 缺失时首次调用报连接错误）。"""
    global _engine
    if _engine is None:
        _engine = create_async_engine(_resolve_dsn(), pool_pre_ping=True, echo=False)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """单例 sessionmaker（绑定 engine）。"""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：yield 一个 AsyncSession（自动关闭）。"""
    async with get_session_factory()() as session:
        yield session


async def init_db() -> None:
    """启动建表（Base.metadata.create_all，幂等）。仅创建缺失表，不改既有表。

    导入 app.models 以确保所有已在 Base.metadata 注册的模型（如 User）被创建。
    """
    import app.models  # noqa: F401  — 注册全部表模型

    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
