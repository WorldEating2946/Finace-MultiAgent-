"""
app/api/auth.py — 认证路由

    POST /api/v1/auth/login     — 用户名/邮箱 + 密码 → access+refresh 双 token
    POST /api/v1/auth/refresh   — refresh token → 新 pair（轮换）
    POST /api/v1/auth/logout    — 登出（bump token_version 吊销该用户所有 token）
    GET  /api/v1/auth/me        — 当前用户信息（Bearer access）
    GET  /api/v1/auth/users     — 用户列表（仅 admin）

登录/刷新开放；me/logout 需 access token；users 需 admin。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_token
from app.database.session import get_db
from app.models.user import User
from app.schemas.auth import LoginRequest, RefreshRequest, TokenPair, UserOut
from app.services.auth import (
    authenticate,
    get_current_user,
    issue_token_pair,
    require_admin,
    revoke_user_tokens,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/login", response_model=TokenPair)
async def login(
    req: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenPair:
    """登录：identifier 支持用户名或邮箱。"""
    user = await authenticate(db, req.identifier, req.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")

    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(user)

    return TokenPair(**issue_token_pair(user), user=UserOut.model_validate(user))


@router.post("/refresh", response_model=TokenPair)
async def refresh(
    req: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenPair:
    """用 refresh token 换取新 access+refresh（轮换）。"""
    payload = decode_token(req.refresh_token, expected_type="refresh")
    if payload is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "refresh token 无效或已过期")

    user = await db.get(User, int(payload["sub"]))
    if user is None or not user.is_active or payload.get("ver") != user.token_version:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "refresh token 已失效")

    return TokenPair(**issue_token_pair(user), user=UserOut.model_validate(user))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """登出：bump token_version，吊销该用户全部 token。"""
    await revoke_user_tokens(db, current_user)


@router.get("/me", response_model=UserOut)
async def me(current_user: Annotated[User, Depends(get_current_user)]) -> UserOut:
    """当前用户信息。"""
    return UserOut.model_validate(current_user)


@router.get("/users", response_model=list[UserOut])
async def list_users(
    _: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[UserOut]:
    """用户列表（仅 admin）。"""
    rows = (await db.execute(select(User).order_by(User.id))).scalars().all()
    return [UserOut.model_validate(u) for u in rows]
