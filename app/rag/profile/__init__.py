"""企业知识画像（PR #34）—— Enterprise Knowledge Profile Layer。

把 Document Knowledge 结构化升级为 Enterprise Model：
    每个字段 = value + description + evidence（source/chapter/section/page/quote）。

    from app.rag.profile import build_profile, save_profile, load_profile
"""

from app.rag.profile.extractor import ProfileExtractor, build_profile
from app.rag.profile.schema import CompanyProfile, EvidenceRef, ProfileItem
from app.rag.profile.storage import load_profile, save_profile

__all__ = [
    "CompanyProfile",
    "ProfileItem",
    "EvidenceRef",
    "ProfileExtractor",
    "build_profile",
    "save_profile",
    "load_profile",
]
