"""Research Intent Schema（PR #36）。

把自然语言研究请求结构化为：
    意图（ResearchIntent） + 目标（ResearchTarget） + 维度（ResearchDimension）
    → 有序 ResearchStep 序列（ResearchPlan）。

本模块是纯规划层——不执行任何检索/抽取，只做意图理解与步骤规划。
执行在 PR #37 Research Agent Workflow。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ResearchIntent(str, Enum):
    """金融研究分析意图分类（规则分类依据，见 intent.py）。"""

    COMPETITIVE_ANALYSIS = "competitive_analysis"     # 竞争力分析
    BUSINESS_OVERVIEW = "business_overview"           # 企业概况/商业模式
    FINANCIAL_ANALYSIS = "financial_analysis"         # 财务分析
    RISK_ANALYSIS = "risk_analysis"                   # 风险分析
    STRATEGY_ANALYSIS = "strategy_analysis"           # 战略分析/未来方向
    MARKET_ANALYSIS = "market_analysis"               # 市场/行业分析
    POLICY_ANALYSIS = "policy_analysis"               # 政策环境分析
    TECHNOLOGY_ANALYSIS = "technology_analysis"       # 技术/研发分析
    GENERIC_RESEARCH = "generic_research"             # 兜底综合研究


class ResearchTarget(BaseModel):
    """分析目标实体。"""

    company: str              # 目标公司（"小米"/"宁德时代"/...）
    segment: str = ""         # 业务板块（"汽车"/"手机"/"电池"...），空 = 全公司


class ResearchDimension(str, Enum):
    """研究维度（跨 intent 共享的分析角度）。"""

    BUSINESS = "business"           # 商业模式/业务矩阵
    TECHNOLOGY = "technology"       # 技术/研发
    MARKET = "market"               # 市场/客户
    FINANCIAL = "financial"         # 财务
    COMPETITION = "competition"     # 竞争/壁垒
    POLICY = "policy"               # 政策/监管
    RISK = "risk"                   # 风险
    STRATEGY = "strategy"           # 战略/未来
    PRODUCT = "product"             # 产品/矩阵


class ResearchStep(BaseModel):
    """单个研究步骤——PR #37 Agent 将要执行的最小单元。"""

    order: int                       # 执行顺序（1-based）
    name: str                        # "企业知识画像"
    description: str = ""            # 该步骤做什么
    retrieval_query: str             # 实际检索 query（{company}/{segment} 已替换）
    dimensions: list[str] = Field(default_factory=list)      # 涉及 ResearchDimension 值
    source_types: list[str] = Field(default_factory=list)    # 涉及数据源
    profile_fields: list[str] = Field(default_factory=list)  # 预期填充 Profile 维度
    depends_on: list[int] = Field(default_factory=list)      # 前置步骤 order


class ResearchPlan(BaseModel):
    """结构化研究计划——PR #36 最终输出。"""

    request: str                       # 原始自然语言请求
    intent: ResearchIntent
    target: ResearchTarget
    dimensions: list[ResearchDimension] = Field(default_factory=list)
    steps: list[ResearchStep] = Field(default_factory=list)
    confidence: float = 0.0            # 意图理解置信度 [0, 1]
    created_at: str = ""               # ISO 8601
