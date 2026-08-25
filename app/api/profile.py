"""企业画像端点（PR40，薄封装 load_profile()）。

内部依赖 get_profile_loader()（测试可注入 temp 文件 loader，hermetic）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.exceptions import ResearchNotFound
from app.core.response import ok

router = APIRouter(prefix="/profile", tags=["profile"])


def get_profile_loader():
    """依赖：画像加载函数（默认 load_profile；测试注入 mock）。"""
    from app.rag.profile.storage import load_profile

    return load_profile


@router.get("/{company}")
def get_profile(company: str, loader=Depends(get_profile_loader)) -> dict:
    """读取企业画像（data/profiles/<company>.json）。"""
    profile = loader(company)
    if profile is None:
        raise ResearchNotFound(f"profile not found for company: {company}")
    return ok(profile.model_dump())
