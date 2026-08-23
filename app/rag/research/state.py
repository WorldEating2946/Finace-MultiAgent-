"""研究执行状态（PR #37）。

ResearchState 是 Executor 的可序列化中间表示（IR）：
    profile + findings（每步证据）+ evidence_pool（去重证据池）。
报告生成只消费 evidence_pool，最终报告可审计。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.rag.profile.schema import CompanyProfile, EvidenceRef
from app.rag.research.schema import ResearchPlan, ResearchTarget


class Finding(BaseModel):
    """单个步骤的执行结果——该步骤收集到的证据。"""

    step_order: int
    step_name: str
    claim: str = ""                       # 短总结（报告合成时由 LLM 提炼）
    evidence: list[EvidenceRef] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)


class ResearchState(BaseModel):
    """研究执行状态（可序列化、可恢复，LangGraph 可直接迁移）。"""

    request: str
    intent: str                           # ResearchIntent.value
    target: ResearchTarget
    plan: ResearchPlan                    # 完整研究计划
    profile: CompanyProfile | None = None          # 企业画像（画像步骤填充）
    findings: list[Finding] = Field(default_factory=list)          # 每步收集
    evidence_pool: list[EvidenceRef] = Field(default_factory=list)  # 去重聚合
    completed_steps: list[int] = Field(default_factory=list)        # 已完成的 step order
    started_at: str = ""
    updated_at: str = ""
