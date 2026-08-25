"""系统就绪状态 —— 前端可见各 API/依赖可用情况。

    GET /api/v1/health/status   （需登录）

逐项探测：database / milvus / akshare / embedding / reranker，
返回每项 {name, ok, detail, ms} + overall，供前端"系统状态"面板展示。
只读、防御式（单项失败不影响整体），异常以 ok=False + detail 呈现而非抛错。
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.response import ok
from app.models.user import User
from app.services.auth import get_current_user

router = APIRouter(prefix="/api/v1/health", tags=["health"])


async def _check(name: str, fn, is_async: bool = False) -> dict:
    t = time.time()
    try:
        detail = await fn() if is_async else await asyncio.to_thread(fn)
        return {"name": name, "ok": True, "detail": detail or "ok", "ms": round((time.time() - t) * 1000)}
    except Exception as exc:  # noqa: BLE001 —— 单项失败仅上报，不中断
        return {"name": name, "ok": False, "detail": str(exc)[:200], "ms": round((time.time() - t) * 1000)}


async def _check_db() -> str:
    from sqlalchemy import text

    from app.database.session import get_engine

    async with get_engine().connect() as conn:
        await conn.execute(text("SELECT 1"))
    return "已连接"


def _check_milvus() -> str:
    from pymilvus import MilvusClient

    from app.core.config import settings

    c = MilvusClient(uri=settings.milvus_uri)
    ver = c.get_server_version()
    if settings.milvus_db_name not in c.list_databases():
        raise RuntimeError(f"缺少数据库: {settings.milvus_db_name}")
    c2 = MilvusClient(uri=settings.milvus_uri, db_name=settings.milvus_db_name)
    if not c2.has_collection(settings.milvus_collection_name):
        raise RuntimeError(f"缺少 collection: {settings.milvus_collection_name}")
    return f"已就绪 v{ver}"


def _check_akshare() -> str:
    import akshare

    return f"已安装 v{getattr(akshare, '__version__', '?')}"


def _check_embedding() -> str:
    from app.rag.embedding import get_embedding_model

    m = get_embedding_model()
    dev = getattr(m, "_resolve_device", lambda: "?")()
    loaded = getattr(m, "_model", None) is not None
    return f"dim={m.dim} device={dev} {'已加载' if loaded else '未加载(首次用时加载)'}"


def _check_reranker() -> str:
    from app.rag.reranker import get_reranker

    return type(get_reranker()).__name__


def _check_finbert() -> str:
    from app.tools.sentiment_tools import _finbert_cached

    if _finbert_cached():
        return "已缓存可加载"
    return "未缓存(HuggingFace 不可达 → 评分降级 NEUTRAL)"


@router.get("/status")
async def system_status(_: Annotated[User, Depends(get_current_user)]) -> dict:
    """返回各项依赖的就绪状态（前端系统状态面板用）。"""
    checks = await asyncio.gather(
        _check("database", _check_db, is_async=True),
        _check("milvus", _check_milvus),
        _check("akshare", _check_akshare),
        _check("embedding", _check_embedding),
        _check("reranker", _check_reranker),
        _check("finbert", _check_finbert),
    )
    return ok({
        "overall": all(c["ok"] for c in checks),
        "checked_at": datetime.now().isoformat(),
        "services": list(checks),
    })
