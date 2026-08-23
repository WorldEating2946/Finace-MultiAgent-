"""
scripts/seed_users.py — 注入认证用户（幂等）

用法（conda finance-agent 环境）:
    python scripts/seed_users.py

行为：
    - 首次启动建表（users；create_all 幂等）
    - 按 DEFAULT_USERS 幂等注入 admin/analyst/demo 三个用户：
        已存在 → 更新字段（email/full_name/is_admin/密码），不存在 → 新增
    - 仅操作 users 表，绝不触碰既有表 / Milvus default 库

生产必读：默认密码仅作演示，部署前务必修改或改用环境变量注入。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 保证可 import app 包（脚本位于 scripts/ 子目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.core.security import hash_password
from app.database.session import get_session_factory, init_db
from app.models.user import User

# (username, email, password, full_name, is_admin)
DEFAULT_USERS: list[tuple[str, str, str, str, bool]] = [
    ("admin", "admin@finaceagent.local", "admin123", "系统管理员", True),
    ("analyst", "analyst@finaceagent.local", "analyst123", "高级分析师", False),
    ("demo", "demo@finaceagent.local", "demo123", "演示用户", False),
]


async def seed(users: list[tuple[str, str, str, str, bool]] = DEFAULT_USERS) -> None:
    """幂等注入用户。"""
    await init_db()
    factory = get_session_factory()
    async with factory() as session:
        for username, email, password, full_name, is_admin in users:
            existing = (
                await session.execute(select(User).where(User.username == username))
            ).scalar_one_or_none()

            if existing:
                existing.email = email
                existing.full_name = full_name
                existing.is_admin = is_admin
                existing.hashed_password = hash_password(password)
                print(f"更新用户: {username}")
            else:
                session.add(
                    User(
                        username=username,
                        email=email,
                        hashed_password=hash_password(password),
                        full_name=full_name,
                        is_admin=is_admin,
                        is_active=True,
                    )
                )
                print(f"新增用户: {username}")
        await session.commit()
    print("用户注入完成")


if __name__ == "__main__":
    asyncio.run(seed())
