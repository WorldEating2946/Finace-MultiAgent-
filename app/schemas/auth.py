"""
app/schemas/auth.py — 认证请求/响应 DTO
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserOut(BaseModel):
    """对外返回的用户信息（不含密码哈希）。"""

    model_config = ConfigDict(from_attributes=True)  # 可直接 from ORM User

    id: int
    username: str
    email: str
    full_name: str | None = None
    is_active: bool
    is_admin: bool
    created_at: datetime


class LoginRequest(BaseModel):
    """登录：identifier 支持用户名或邮箱。"""

    identifier: str = Field(..., min_length=1, description="用户名或邮箱")
    password: str = Field(..., min_length=1, description="明文密码")


class RefreshRequest(BaseModel):
    """用 refresh token 换取新 access/refresh。"""

    refresh_token: str = Field(..., min_length=1, description="refresh token")


class TokenPair(BaseModel):
    """登录/刷新成功返回的双 token + 用户信息。"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # access token 剩余秒数（前端用它控制自动刷新）
    user: UserOut
