"""多源融合 Schema（PR #35）。

SourceType：文档语义类型枚举（与 doc_type 格式级分离）。
SourceConflict：跨源证据冲突——两个来源对同一实体给出矛盾描述时记录。
EnterpriseKnowledgePackage：多源融合的最终输出。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.rag.profile.schema import CompanyProfile, EvidenceRef


class SourceType(str, Enum):
    """文档语义类型（PR #35 多源融合）。

    与 doc_type（格式级：pdf/markdown/text，splitter 依赖）分离：
        source_type 是语义级，标注文档在投资研究中的角色——
        年报（公司自述）、研报（市场观点）、政策（外部环境）、新闻（近期变化）。
    """

    ANNUAL_REPORT = "annual_report"
    RESEARCH_REPORT = "research_report"
    POLICY = "policy"
    NEWS = "news"


# 画像融合涉及的维度（CompanyProfile 中 list[ProfileItem] 的字段）
FUSION_DIMENSIONS = [
    "business_segments",
    "products",
    "technologies",
    "customers",
    "geographic_markets",
    "competitive_advantages",
    "risks",
    "strategic_direction",
]


class SourceConflict(BaseModel):
    """跨源证据冲突——不同来源对同一实体给出矛盾描述时记录。

    不自动裁决：Research Agent 依据 claim_a/claim_b 与各自证据链做判断。
    """

    dimension: str                      # "business_segments" / "strategic_direction" 等
    entity_a: str                       # 来源 A 的实体名
    entity_b: str                       # 来源 B 的实体名（命名可能不同）
    claim_a: str                        # 来源 A 的描述
    claim_b: str                        # 来源 B 的描述
    source_a: str                       # 来源 A 的 source_type
    source_b: str                       # 来源 B 的 source_type
    evidence_a: EvidenceRef | None = None   # 来源 A 的证据引用（可溯源）
    evidence_b: EvidenceRef | None = None   # 来源 B 的证据引用
    resolution_note: str = ""           # 人类可读的差异说明（预留 Research Agent 填充）


class EnterpriseKnowledgePackage(BaseModel):
    """多源企业知识融合包——PR #35 核心输出。

    company_name + 各源独立画像 + 融合画像 + 跨源冲突 + 证据统计。
    """

    company_name: str
    profiles: dict[str, CompanyProfile]      # source_type → 该源独立画像
    fused: CompanyProfile                    # 融合后的统一画像（多源确认实体合并证据）
    conflicts: list[SourceConflict] = Field(default_factory=list)  # 跨源冲突列表
    evidence_summary: dict[str, int]         # source_type → 该源证据条目数
    extracted_at: str = ""                   # ISO 8601 时间戳
