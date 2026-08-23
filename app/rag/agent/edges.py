"""LangGraph 条件边（PR38+PR40）。

router_fn      —— evaluate 后按 next_action 分流：
                   replan + human_review → review（人工闸口）；replan（默认）→ 直接补步；end → 终止
human_router_fn —— review 后按人工决策分流：reject → 终止；approve/modify → 补步继续
"""

from __future__ import annotations

from app.rag.agent.state import AgentState


def router_fn(state: AgentState) -> str:
    """条件边：evaluate 后分流（replan / review / end）。"""
    if state.next_action == "replan":
        return "review" if state.human_review else "replan"
    return "end"


def human_router_fn(state: AgentState) -> str:
    """条件边：review 后按人工决策分流（reject → end，approve/modify → replan）。"""
    decision = state.human_decision or {}
    action = decision.get("action", "approve")
    return "reject" if action == "reject" else "approve"
