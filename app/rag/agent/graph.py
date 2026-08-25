"""自适应研究 LangGraph（PR38）。

StateGraph 用 AgentState（Pydantic）作 state schema：
    START → intent → planning → execute → report → evaluate
        → [router] → replan → execute（循环，最多 _MAX_ITERATIONS 轮）
        → END

invoke(AgentState) → dict（LangGraph 序列化）；run_adaptive_research 负责还原 AgentState。
依赖注入：build_graph(tools=..., report_builder=...) —— 测试注入 mock，零 LLM / 向量库。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.rag.agent.edges import human_router_fn, router_fn
from app.rag.agent.nodes import build_nodes
from app.rag.agent.state import AgentState


def build_graph(*, tools=None, report_builder=None, checkpointer=None):
    """构建并编译自适应研究 StateGraph（state schema = AgentState）。

    Args:
        tools:         ResearchTools（测试注入 mock；默认真实工具）。
        report_builder: ReportBuilder（测试注入 mock；默认真实 LLM 合成）。
        checkpointer:  LangGraph checkpointer（MemorySaver/SqliteSaver，PR39 持久化）。

    Returns:
        CompiledStateGraph —— invoke(AgentState.from_request(req), config=...) → dict。
    """
    nodes = build_nodes(tools=tools, report_builder=report_builder)
    graph = StateGraph(AgentState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.add_edge(START, "intent")
    graph.add_edge("intent", "planning")
    graph.add_edge("planning", "execute")
    graph.add_edge("execute", "report")
    graph.add_edge("report", "evaluate")
    graph.add_conditional_edges(
        "evaluate", router_fn, {"replan": "replan", "review": "review", "end": END}
    )
    graph.add_conditional_edges(
        "review", human_router_fn, {"approve": "replan", "reject": END}
    )
    graph.add_edge("replan", "execute")

    return graph.compile(checkpointer=checkpointer)
