"""
app/agents/financial_agent — Financial Agent 子模块

Financial Agent 负责:
    1. 异步获取原始财务数据（通过 MarketDataService）
    2. 硬计算财务指标（通过 FinancialCalculator）
    3. 调用 LLM 生成资深 CFO 视角的财务健康度点评
    4. 将结构化结果写入 LangGraph 工作流状态

供主图调用的入口:
    from app.agents.financial_agent.node import financial_analysis_node

    builder.add_node("financial", financial_analysis_node)
"""

from app.agents.financial_agent.node import financial_analysis_node
from app.agents.financial_agent.schemas import FinancialAgentInput, FinancialAgentOutput

__all__ = [
    "FinancialAgentInput",
    "FinancialAgentOutput",
    "financial_analysis_node",
]
