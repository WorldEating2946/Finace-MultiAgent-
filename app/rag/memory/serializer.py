"""Research Memory 序列化（PR39）—— Memory Boundary。

定义"什么该持久化、什么该重新生成"：
    ResearchRecord = AgentState 的压缩衍生视图（human view / 任务状态）。

必须持久化（Record 字段）：
    任务信息（research_id/query/company/intent/created_at）、执行进度（current_step/completed_steps）、
    Agent 决策（iteration/next_action/missing_dimensions）、中间产物摘要（evidence_count/finding_count/coverage）。

不持久化（应重新生成）：
    Embedding vector / LLM raw response / model cache / debug info。
    → 这些本就不在 AgentState（PR37 设计的可序列化 IR），天然满足边界。
"""

from __future__ import annotations

from app.rag.agent.state import AgentState
from app.rag.memory.schema import RecordStatus, ResearchRecord


def derive_status(state: AgentState, *, pending_writes: bool = False) -> RecordStatus:
    """从 checkpoint 状态推导任务状态。

    - pending interrupt 写入 → PAUSED（中断 / 等人工审核）
    - human_decision.action=="reject" → REJECTED（PR42a：被拒绝任务独立状态，先于 end 判定）
    - 无 pending 且 next_action="end"（Agent 已决策终止）→ COMPLETED
    - 否则 → RUNNING
    """
    if pending_writes:
        return RecordStatus.PAUSED
    if (state.human_decision or {}).get("action") == "reject":
        return RecordStatus.REJECTED
    if state.next_action == "end":
        return RecordStatus.COMPLETED
    return RecordStatus.RUNNING


def to_record(state: AgentState, research_id: str, *, pending_writes: bool = False) -> ResearchRecord:
    """AgentState → ResearchRecord（Memory Boundary 压缩视图）。

    Args:
        state:           checkpoint 读取的最新 AgentState。
        research_id:     thread_id（LangGraph 线程唯一标识）。
        pending_writes:  checkpoint 是否有 pending interrupt（用于 PAUSED 判定）。

    Returns:
        ResearchRecord（任务信息 + 进度 + 决策 + 产物摘要）。
    """
    by_order = {s.order: s for s in state.plan.steps}
    completed_names = [
        by_order[order].name for order in state.completed_steps if order in by_order
    ]
    return ResearchRecord(
        research_id=research_id,
        query=state.request,
        company=state.target.company,
        intent=state.intent,
        created_at=state.started_at,
        updated_at=state.updated_at,
        status=derive_status(state, pending_writes=pending_writes),
        current_step=state.current_step,
        completed_steps=completed_names,
        iteration=state.iteration,
        next_action=state.next_action,
        missing_dimensions=list(state.missing_dimensions),
        evidence_count=len(state.evidence_pool),
        finding_count=len(state.findings),
        coverage=state.evaluation.evidence_coverage if state.evaluation else None,
    )
