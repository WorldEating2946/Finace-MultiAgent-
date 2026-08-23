"""自适应研究 Agent（PR38）—— Research Loop。

不是"接入 LangGraph"，而是让 Agent 根据研究质量自动决定：继续查、补什么、什么时候结束。

    Plan → Execute → Report → Evaluate → [补步 → 循环] → 结果

最多 _MAX_ITERATIONS 轮（第 3 轮强制输出），基于 PR37.5 质量指标决策：
    - 证据缺失维度 → 动态补步（planner.add_replan_step）
    - 质量达标 / 迭代耗尽 → 终止

    from app.rag.agent import run_adaptive_research
    state = run_adaptive_research("分析小米汽车未来竞争力")
    # state.iteration / state.evaluation / state.current_report / state.next_action
"""

from __future__ import annotations

from app.rag.agent.graph import build_graph
from app.rag.agent.state import AgentState


def run_adaptive_research(
    request: str,
    *,
    thread_id: str | None = None,
    checkpointer=None,
    human_review: bool = False,
    max_iterations: int = 3,
    _tools=None,
    _report_builder=None,
) -> AgentState:
    """自适应研究端到端（同步）：NL 请求 → 最多 3 轮研究 → AgentState（含最终报告）。

    Args:
        request:        自然语言研究请求（"分析小米汽车未来竞争力"）。
        thread_id:      任务唯一标识（checkpointer 存在时，LangGraph config thread_id）。
        checkpointer:   LangGraph checkpointer（MemorySaver/SqliteSaver，PR39 持久化）。
        human_review:   PR40 —— True 时证据不足需补步前暂停，等待人工审核决策。
        _tools:         测试 seam —— 注入 ResearchTools mock。
        _report_builder: 测试 seam —— 注入 ReportBuilder mock。

    Returns:
        AgentState（含 evaluation / current_report / iteration / next_action）。
        若在 review_node 暂停（human_review 中断），返回中断前的状态。
    """
    graph = build_graph(tools=_tools, report_builder=_report_builder, checkpointer=checkpointer)
    config = {"thread_id": thread_id} if thread_id else None
    initial = AgentState.from_request(request)
    over: dict = {
        "max_iterations": max_iterations,
        "max_evidence": 20 if max_iterations == 1 else 60,
        "max_steps": 3 if max_iterations == 1 else 10,
    }
    if human_review:
        over["human_review"] = True
    initial = initial.model_copy(update=over)
    result = graph.invoke(initial, config=config)
    return _to_agent_state(result)


async def arun_adaptive_research(
    request: str,
    *,
    thread_id: str | None = None,
    checkpointer=None,
    human_review: bool = False,
    max_iterations: int = 3,
    _tools=None,
    _report_builder=None,
    event_sink=None,
) -> AgentState:
    """自适应研究端到端（异步，PR41）：graph.ainvoke，生产 Runtime 路径。

    与 run_adaptive_research 相同，但使用 ainvoke（配合 AsyncPostgresSaver /
    MemorySaver 均可）。event_sink 为可选的 async 回调：
        await event_sink({"type": "node_start", "node": "execute", ...})
    用于 SSE 事件流（PR41.3），每节点开始/结束推送。

    Args:
        request:         自然语言研究请求。
        thread_id:       任务唯一标识。
        checkpointer:    LangGraph checkpointer（AsyncPostgresSaver / SqliteSaver）。
        human_review:    证据不足需补步前暂停，等待人工审核。
        _tools:          测试 seam —— ResearchTools mock。
        _report_builder: 测试 seam —— ReportBuilder mock。
        event_sink:      异步事件回调（PR41.3 SSE；None 时跳过事件推送）。

    Returns:
        AgentState（含 evaluation / current_report / iteration / next_action）。
    """
    graph = build_graph(tools=_tools, report_builder=_report_builder, checkpointer=checkpointer)
    config = {"thread_id": thread_id} if thread_id else None
    initial = AgentState.from_request(request)
    over: dict = {
        "max_iterations": max_iterations,
        "max_evidence": 20 if max_iterations == 1 else 60,
        "max_steps": 3 if max_iterations == 1 else 10,
    }
    if human_review:
        over["human_review"] = True
    initial = initial.model_copy(update=over)

    if event_sink is not None:
        result = await _ainvoke_with_events(graph, initial, config, event_sink)
    else:
        result = await graph.ainvoke(initial, config=config)
    return _to_agent_state(result)


async def _ainvoke_with_events(graph, initial, config, event_sink):
    """用 graph.astream_events 驱动执行，逐节点推送事件（PR41.3 SSE 数据源）。

    astream_events(version="v2") 在 Pydantic-schema StateGraph 上可用；
    on_chain_start / on_chain_end 事件对应节点进入/退出。
    """
    final: dict | None = None
    async for event in graph.astream_events(initial, config=config, version="v2"):
        kind = event.get("event", "")
        name = event.get("name", "")
        if kind == "on_chain_start":
            await event_sink({
                "type": "node_start",
                "node": name,
                "message": f"开始节点：{name}",
            })
        elif kind == "on_chain_end":
            await event_sink({
                "type": "node_end",
                "node": name,
                "message": f"完成节点：{name}",
            })
            if name == "LangGraph" and "data" in event:
                out = event.get("data", {}).get("output")
                if isinstance(out, dict):
                    final = out
    # astream_events 不直接暴露终态 dict → 用返回的 AgentState 兜底
    if final is None:
        # 重新 invoke 一次拿终态（幂等，checkpoint 命中）
        result = await graph.ainvoke(None, config=config)
        if isinstance(result, AgentState):
            return result
        final = dict(result)
    return final


def _to_agent_state(result) -> AgentState:
    """LangGraph Pydantic state：invoke 返回 dict（可能含 __interrupt__）→ 还原为 AgentState。"""
    if isinstance(result, AgentState):
        return result
    data = dict(result)
    data.pop("__interrupt__", None)  # interrupt 不是 AgentState 字段，剥离
    return AgentState(**data)


__all__ = ["AgentState", "build_graph", "run_adaptive_research", "arun_adaptive_research"]
