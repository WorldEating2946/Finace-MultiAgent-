"""
app/core/security.py — JWT 双 Token + 密码哈希

双 Token 设计：
    access  — 短命（默认 15min），Bearer 头传，前端 401 时用 refresh 续签
    refresh — 长命（默认 7d），仅用于 /auth/refresh 换取新 access

payload 统一含 sub（user_id）、type（access/refresh）、ver（token_version，
服务端吊销杠杆）、iat、exp。verify 时校验 exp + type + 与用户 token_version 匹配。

密钥从 settings.jwt_secret_key（.env 的 JWT_SECRET_KEY）读取。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT payload 中 token 版本声明名（对应 User.token_version）
_TOKEN_VERSION_CLAIM = "ver"


# ── 密码哈希 ──────────────────────────────────────────────
def hash_password(password: str) -> str:
    """明文 → bcrypt 哈希。"""
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """校验明文与存储哈希是否匹配。"""
    return pwd_context.verify(plain, hashed)


# ── Token 签发 ────────────────────────────────────────────
def _create_token(
    subject: str | int,
    *,
    token_type: str,
    ver: int,
    lifetime: timedelta,
    **extra: Any,
) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": token_type,
        _TOKEN_VERSION_CLAIM: ver,
        "iat": now,
        "exp": now + lifetime,
    }
    payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(subject: str | int, ver: int = 0, **extra: Any) -> str:
    """签发 access token（短命）。"""
    return _create_token(
        subject,
        token_type="access",
        ver=ver,
        lifetime=timedelta(minutes=settings.access_token_expire_minutes),
        **extra,
    )


def create_refresh_token(subject: str | int, ver: int = 0, **extra: Any) -> str:
    """签发 refresh token（长命）。"""
    return _create_token(
        subject,
        token_type="refresh",
        ver=ver,
        lifetime=timedelta(days=settings.refresh_token_expire_days),
        **extra,
    )


# ── Token 解析 ────────────────────────────────────────────
def decode_token(token: str, expected_type: str | None = None) -> dict[str, Any] | None:
    """解码并校验 token。

    返回 payload；签名/过期/类型不符 → None（不抛异常，调用方按鉴权失败处理）。
    """
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except jwt.PyJWTError:
        return None
    if expected_type is not None and payload.get("type") != expected_type:
        return None
    return payload
