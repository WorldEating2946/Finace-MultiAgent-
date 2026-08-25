"""人工审核负载构造（PR40 Human-in-the-loop）。

review_node 在 Agent 判定"证据不足需补步"时暂停，向人工展示：
    缺失维度 / 当前进度 / 迭代轮次 / 决策问题。

build_review_payload(state) 把 AgentState 转成人工可读的审核请求。
"""

from __future__ import annotations

from pydantic import BaseModel

from app.rag.agent.state import AgentState


class HumanReviewRequest(BaseModel):
    """展示给人工的审核请求。"""

    missing_dimensions: list[str] = []
    current_step: str = ""
    iteration: int = 0
    question: str = ""          # 人类可读的决策问题


def build_review_payload(state: AgentState) -> HumanReviewRequest:
    """AgentState → 审核请求（缺失维度 + 决策问题）。

    Args:
        state: 评估后、补步前的 AgentState（missing_dimensions 已由 evaluate 推导）。

    Returns:
        HumanReviewRequest —— review_node 的 interrupt payload。
    """
    missing = list(state.missing_dimensions)
    dims = "、".join(missing) if missing else "当前维度"
    return HumanReviewRequest(
        missing_dimensions=missing,
        current_step=state.current_step,
        iteration=state.iteration,
        question=f"以下研究维度证据不足：{dims}，是否继续补充研究？"
        f"（approve 继续 / reject 停止 / modify 带反馈继续）",
    )
