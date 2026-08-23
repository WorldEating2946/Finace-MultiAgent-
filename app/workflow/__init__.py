"""
app/workflow — LangGraph 工作流定义

本模块定义 FinanceAgent 的核心状态机:
    - state: 全链路 TypedDict 状态 (ResearchState)
    - graph: 节点函数 + Send API 并行路由 + 图编译

供外部调用:
    from app.workflow.state import ResearchState
    from app.workflow.graph import build_graph
"""

from app.workflow.graph import build_graph
from app.workflow.state import ResearchState

__all__ = ["ResearchState", "build_graph"]
