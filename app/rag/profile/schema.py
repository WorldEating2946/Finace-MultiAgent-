"""企业知识画像 Schema（PR #34）。

把 Document Knowledge 结构化升级为 Enterprise Model：
    每个画像字段 = value + description + evidence（source/chapter/section/page/quote）。

关键设计（抗幻觉归因）：
    - 不存原始文本片段，存结构化实体；
    - 每个实体必须带证据引用（EvidenceRef），可追溯到年报原文；
    - EvidenceRef.quote 是从原文 chunk 提取的引用（≤200 字符）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceRef(BaseModel):
    """证据引用：该字段值的可溯来源。"""

    source: str                    # "小米集团2025年报.pdf"
    source_type: str = ""          # 文档语义类型（PR #35）：annual_report / research_report / policy / news
    chapter: str = ""              # 章名（如 "管理层讨论及分析"）
    section: str = ""              # 小节路径（如 "5.1 智能电动汽车业务"）
    page: int | None = None        # 页码
    quote: str = ""                # 原文引用（≤200 字符）
    chunk_id: str = ""             # 来源 chunk 唯一标识（可回溯向量库）


class ProfileItem(BaseModel):
    """画像维度下的单条实体（含证据链）。"""

    name: str                      # 实体名（如 "智能手机"）
    description: str = ""          # 一句话描述
    evidence: list[EvidenceRef] = Field(default_factory=list)  # 该值的证据来源


class CompanyProfile(BaseModel):
    """企业知识画像（Enterprise Model）。"""

    company_name: str
    industry: str = ""                        # 所属行业（如 "智能硬件与消费电子"）
    business_segments: list[ProfileItem] = Field(default_factory=list)   # 业务矩阵
    products: list[ProfileItem] = Field(default_factory=list)            # 产品矩阵
    technologies: list[ProfileItem] = Field(default_factory=list)        # 核心技术
    customers: list[ProfileItem] = Field(default_factory=list)           # 客户/市场
    geographic_markets: list[ProfileItem] = Field(default_factory=list)  # 地理市场
    competitive_advantages: list[ProfileItem] = Field(default_factory=list)  # 竞争优势
    risks: list[ProfileItem] = Field(default_factory=list)               # 风险因素
    strategic_direction: list[ProfileItem] = Field(default_factory=list) # 战略方向
    extracted_at: str = ""        # ISO 8601 时间戳
