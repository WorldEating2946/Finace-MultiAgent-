"""企业画像持久化（PR #34）。

存储为 JSON 文件（含缩进，可人工审计证据链）。不做数据库/图谱 —— 单企业画像
规模小，JSON + Evidence 足够，后续多公司竞争分析再考虑图谱化。

路径：data/profiles/<company>.json（公司名可含中文，Python 原生处理 Unicode 路径）。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.rag.profile.schema import CompanyProfile

_DEFAULT_DIR = Path("data") / "profiles"


def _profile_path(company: str, base: Path = _DEFAULT_DIR) -> Path:
    return base / f"{company}.json"


def save_profile(profile: CompanyProfile, path: str | None = None) -> Path:
    """保存画像为 JSON（缩进 2，ensure_ascii=False 保留中文）。"""
    target = Path(path) if path else _profile_path(profile.company_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        profile.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def load_profile(company: str, path: str | None = None) -> CompanyProfile | None:
    """加载画像；文件不存在返回 None。"""
    target = Path(path) if path else _profile_path(company)
    if not target.exists():
        return None
    return CompanyProfile.model_validate_json(target.read_text(encoding="utf-8"))
