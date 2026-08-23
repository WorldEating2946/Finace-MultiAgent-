"""统一响应信封（PR40）。

所有 API 返回 {code, message, data}：
    - code=0     成功
    - code≠0     业务错误（见 exceptions.py 的 AppError.code）
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import AppError


def ok(data: Any = None, message: str = "ok") -> dict:
    """成功响应。"""
    return {"code": 0, "message": message, "data": data}


def error(e: AppError) -> dict:
    """错误响应（由 exception_handler 调用）。"""
    return {"code": e.code, "message": e.message, "data": None}
