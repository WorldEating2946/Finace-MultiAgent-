"""Human-in-the-loop 决策模型（PR40）。

HumanDecision 是人工审核闸口（review_node）的决策载荷：
    - action: approve（继续补充研究）/ reject（停止）/ modify（带反馈继续）
    - feedback: 审核备注（审计留痕）

review_node 的 interrupt() 收到的是 resume 时传入的 dict（LangGraph Command(resume=...)），
API 层把用户请求体校验为 HumanDecision 后透传。
"""

from __future__ import annotations

from pydantic import BaseModel

# 合法决策动作
_VALID_ACTIONS = ("approve", "reject", "modify")


class HumanDecision(BaseModel):
    """人工审核决策。"""

    action: str = "approve"    # approve | reject | modify
    feedback: str = ""         # 审核备注


def validate_decision(data: dict) -> HumanDecision | None:
    """校验用户输入的决策 dict；非法 action 返回 None。

    Args:
        data: API 层收到的 {action, feedback}。

    Returns:
        合法 → HumanDecision；非法 / 缺字段 → None。
    """
    if not isinstance(data, dict):
        return None
    action = str(data.get("action", "")).strip()
    if action not in _VALID_ACTIONS:
        return None
    return HumanDecision(action=action, feedback=str(data.get("feedback", "") or ""))
