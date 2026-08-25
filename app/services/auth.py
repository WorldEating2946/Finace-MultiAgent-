"""
app/services/auth.py — 认证业务 + FastAPI 依赖

职责：
    authenticate      — 用户名/邮箱 + 密码 → User
    issue_token_pair  — 签发 access+refresh 双 token
    rotate_refresh    — 刷新 token → 新 pair
    get_current_user  — 受保护端点的鉴权依赖（Bearer access）
    require_admin     — 管理端点的角色守卫
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from app.database.session import get_db
from app.models.user import User

# auto_error=False：允许缺失凭据，由下方统一返回 401（而非 403 的 WWW-Authenticate）
_bearer = HTTPBearer(auto_error=False)


async def authenticate(
    db: AsyncSession, identifier: str, password: str
) -> User | None:
    """按用户名或邮箱查找用户并校验密码；失败返回 None。"""
    stmt = select(User).where(
        or_(User.username == identifier, User.email == identifier)
    )
    user = (await db.execute(stmt)).scalar_one_or_none()
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user


def issue_token_pair(user: User) -> dict[str, object]:
    """签发 access+refresh 双 token。ver 取自 user.token_version（吊销杠杆）。"""
    access = create_access_token(user.id, ver=user.token_version)
    refresh = create_refresh_token(user.id, ver=user.token_version)
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60,
    }


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """受保护端点依赖：解析 Bearer access → 用户。

    校验链：缺失/none → 401；签名或过期无效 → 401；用户禁用/不存在 → 401；
    token 版本与用户不符（已吊销）→ 401。
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "未提供认证 token")

    payload = decode_token(credentials.credentials, expected_type="access")
    if payload is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token 无效或已过期")

    user = await db.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户不存在或已禁用")
    if payload.get("ver") != user.token_version:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token 已失效，请重新登录")

    return user


async def require_admin(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """管理端点守卫：非 admin → 403。"""
    if not current_user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "需要管理员权限")
    return current_user


async def revoke_user_tokens(db: AsyncSession, user: User) -> None:
    """登出吊销：bump token_version，使该用户所有历史 token 立即失效。"""
    user.token_version += 1
    await db.commit()

