"""健康检查端点（PR40）。"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.response import ok

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """服务健康检查。"""
    return ok({"status": "ok"}, message="service healthy")
