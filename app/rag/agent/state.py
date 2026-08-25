"""自适应研究 Agent 状态（PR38）。

复用 PR37 ResearchState（profile/findings/evidence_pool/completed_steps），
扩展自适应循环控制字段：
    evaluation          —— 研究报告质量（PR37.5 evaluate_report 输出）
    current_report      —— 最近一轮的 ResearchReport（report_node 产出）
    iteration           —— 已执行研究轮次（execute_node 每轮 +1，上限 _MAX_ITERATIONS）
    missing_dimensions  —— 证据缺失的维度（evaluate_node 推导，replan 候选）
    replanned_dimensions —— 已补充过的维度（replan 去重，防步骤膨胀）
    next_action         —— "continue"(循环中) | "replan"(待补步) | "end"(终止)

LangGraph channel：graph 用单一 channel `research: AgentState`，
每个节点返回 {"research": <新 AgentState>}（model_copy 不可变更新）。
"""

from __future__ import annotations

from pydantic import Field

from app.rag.research.evaluate import ResearchMetrics
from app.rag.research.report import ResearchReport
from app.rag.research.schema import ResearchIntent, ResearchPlan, ResearchTarget
from app.rag.research.state import ResearchState


class AgentState(ResearchState):
    """自适应研究 Agent 状态——ResearchState 的 LangGraph 扩展。"""

    evaluation: ResearchMetrics | None = None
    current_report: ResearchReport | None = None
    iteration: int = 0
    max_iterations: int = 3           # 本轮数上限（快速模式=1，深模式=3；防循环/控成本）
    max_evidence: int = 60            # 报告合成用证据上限（快速模式=20 压缩 LLM 输入→提速）
    max_steps: int = 10               # 单轮执行步骤上限（快速模式=3 减检索次数→提速）
    current_step: str = ""            # 当前执行步骤名（execute_node 逐步骤更新，PR39 进度）
    missing_dimensions: list[str] = Field(default_factory=list)
    replanned_dimensions: list[str] = Field(default_factory=list)
    next_action: str = "continue"
    human_review: bool = False        # 是否开启人工审核闸口（PR40 review_node）
    human_decision: dict = Field(default_factory=dict)  # review_node 收到的人工决策

    @classmethod
    def from_request(cls, request: str) -> "AgentState":
        """初始状态：仅带请求，intent/target/plan 由 intent_node 填充。"""
        return cls(
            request=request,
            intent=ResearchIntent.GENERIC_RESEARCH.value,  # 占位，intent_node 覆盖
            target=ResearchTarget(company=""),
            plan=ResearchPlan(
                request=request,
                intent=ResearchIntent.GENERIC_RESEARCH,
                target=ResearchTarget(company=""),
            ),
        )
