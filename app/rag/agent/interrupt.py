"""LangGraph interrupt 薄封装（PR40 Human-in-the-loop 闸口）。

pause() / resume_command() 是 LangGraph interrupt/Command 的唯一访问点——
PR41 若接入 SSE 事件流 / 持久化通知，只需改本模块，review_node 零改动。
"""

from __future__ import annotations

from langgraph.types import Command, interrupt


def pause(payload: dict):
    """暂停图执行，携带审核请求 payload（等待人工 resume）。

    Args:
        payload: 展示给人工的审核请求（missing_dimensions / question 等）。

    Returns:
        resume 时传入的决策 dict（LangGraph Command(resume=...) 的值）。
    """
    return interrupt(payload)


def resume_command(value: dict) -> Command:
    """构造恢复指令（携带人工决策）。

    Args:
        value: 人工决策 dict（{action, feedback}，见 human.HumanDecision）。

    Returns:
        Command(resume=value) —— 交给 graph.invoke() 继续执行。
    """
    return Command(resume=value)
