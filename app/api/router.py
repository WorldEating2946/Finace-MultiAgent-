"""API 总路由聚合（PR40 + PR41）。

/api/v1 下挂载 5 个子路由：research（研究）/ stream（SSE 事件流）/ knowledge（检索）/
profile（画像）/ health（健康）。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api import health, knowledge, profile, research, stream

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(knowledge.router)
api_router.include_router(profile.router)
api_router.include_router(stream.router)
api_router.include_router(research.router)
