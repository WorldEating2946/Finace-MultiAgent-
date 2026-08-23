"""多源知识融合引擎（PR #35）。

SourceFusion：对每个 source_type 独立构建 CompanyProfile（evidence 带 source_type），
再跨源融合：同名实体合并证据、单源实体保留、冲突实体进 conflicts。
最终输出 EnterpriseKnowledgePackage。

用法：
    from app.rag.source import SourceFusion
    package = SourceFusion("小米", sources=["annual_report", "research_report"]).build()
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.core.config import settings
from app.rag.profile.extractor import ProfileExtractor, build_profile
from app.rag.profile.schema import CompanyProfile, ProfileItem
from app.rag.source.conflict import ConflictDetector, same_entity
from app.rag.source.schema import FUSION_DIMENSIONS, EnterpriseKnowledgePackage

# 融合 description 优先级：年报（公司自述）最高，作为冲突时的基准
_SOURCE_PRIORITY = {"annual_report": 0}


class SourceFusion:
    """多源知识融合引擎：逐源提取 → 跨源融合 → 冲突检测 → KnowledgePackage。"""

    def __init__(
        self,
        company: str,
        sources: list[str] | None = None,
        *,
        _profile_builder: callable | None = None,
    ) -> None:
        """Args:
            company:   目标公司。
            sources:   参与的文档语义类型；None → settings.rag_source_types。
            _profile_builder: 测试 seam —— 注入 (company, source_type) -> CompanyProfile。
        """
        self._company = company
        self._sources = sources or list(settings.rag_source_types)
        self._profile_builder = _profile_builder or self._build_single_source_profile

    def build(self) -> EnterpriseKnowledgePackage:
        """逐源提取 → 跨源融合 → 输出知识包。"""
        profiles: dict[str, CompanyProfile] = {}
        for st in self._sources:
            profiles[st] = self._profile_builder(self._company, st)

        detector = ConflictDetector()
        conflicts = detector.detect(profiles)
        return EnterpriseKnowledgePackage(
            company_name=self._company,
            profiles=profiles,
            fused=self._merge(profiles, conflicts),
            conflicts=conflicts,
            evidence_summary=self._count_evidence(profiles),
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── 单源提取 ────────────────────────────────────────────────
    def _build_single_source_profile(self, company: str, source_type: str) -> CompanyProfile:
        """对指定 source_type 构建独立画像（evidence 带 source_type 标注）。"""
        extractor = ProfileExtractor(company, source_type=source_type)
        return build_profile(company, extractor=extractor)

    # ── 融合 ────────────────────────────────────────────────────
    def _merge(
        self,
        profiles: dict[str, CompanyProfile],
        conflicts: list[SourceConflict],
    ) -> CompanyProfile:
        """跨源融合：同名实体合并证据，冲突实体保留高优先级源版本。

        策略：
            - 同名实体多源出现 → 合并 evidence（跨源确认），description 取优先级最高源；
            - 冲突实体（在 conflicts 中）→ 保留高优先级源（默认年报），差异已在 conflicts 记录；
            - 单源实体 → 原样保留。
        """
        # 冲突实体名集合（用于 fused 中避免低优先级源覆盖高优先级源）
        conflict_names: set[str] = {c.entity_a for c in conflicts} | {
            c.entity_b for c in conflicts
        }

        merged: dict[str, list[ProfileItem]] = {dim: [] for dim in FUSION_DIMENSIONS}
        # 按 source 优先级排序迭代 → existing 总是高优先级源
        ordered = sorted(profiles, key=lambda st: _SOURCE_PRIORITY.get(st, 1))
        for st in ordered:
            for dim in FUSION_DIMENSIONS:
                for item in getattr(profiles[st], dim):
                    existing = self._find_entity(merged[dim], item.name)
                    if existing is None:
                        merged[dim].append(item)
                    else:
                        merged[dim].remove(existing)
                        # 保留 canonical 名（高优先级源命名）+ 合并证据
                        merged[dim].append(
                            ProfileItem(
                                name=existing.name,
                                description=self._pick_description(
                                    existing, item, conflict_names
                                ),
                                evidence=existing.evidence + item.evidence,
                            )
                        )

        industry = self._fused_industry(profiles)
        return CompanyProfile(company_name=self._company, industry=industry, **merged)

    @staticmethod
    def _find_entity(items: list[ProfileItem], name: str) -> ProfileItem | None:
        """在同维度 items 中找同名（含模糊匹配）实体。"""
        for item in items:
            if same_entity(item.name, name):
                return item
        return None

    @staticmethod
    def _pick_description(
        existing: ProfileItem,
        incoming: ProfileItem,
        conflict_names: set[str],
    ) -> str:
        """选 description：冲突实体保持高优先级源（existing）；否则取证据更多的一方。"""
        if existing.name in conflict_names and incoming.name in conflict_names:
            return existing.description
        if len(incoming.evidence) > len(existing.evidence):
            return incoming.description
        return existing.description

    @staticmethod
    def _fused_industry(profiles: dict[str, CompanyProfile]) -> str:
        """fused 行业：年报优先，否则取第一个非空。"""
        if "annual_report" in profiles and profiles["annual_report"].industry:
            return profiles["annual_report"].industry
        for profile in profiles.values():
            if profile.industry:
                return profile.industry
        return ""

    @staticmethod
    def _count_evidence(profiles: dict[str, CompanyProfile]) -> dict[str, int]:
        """各源证据条目数：sum(len(item.evidence))。"""
        summary: dict[str, int] = {}
        for st, profile in profiles.items():
            total = sum(
                len(item.evidence)
                for dim in FUSION_DIMENSIONS
                for item in getattr(profile, dim)
            )
            summary[st] = total
        return summary
