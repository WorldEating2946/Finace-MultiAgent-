"""
app/agents — Agent 业务逻辑层

每个 Agent 为一个独立子模块，封装该 Agent 的:
    - schemas: Pydantic I/O 模型
    - prompts: System / User Prompt 模板
    - node: 异步节点函数（供 LangGraph 主图调用）

已实现:
    - financial_agent: 财务分析 Agent (Member 3)
    - sentiment_agent: 舆情分析 Agent (Member 4)
    - risk_agent: 风险评估 Agent (Member 4)

待实现:
    - research_agent: 企业知识 RAG Agent (Member 2)
    - report_agent: 研报生成 Agent
    - manager_agent: 调度 Agent (Member 1)
"""

# Sentiment & Risk Agent（单文件 Agent 类，Member 4）
from .risk_agent import RiskAgent
from .sentiment_agent import SentimentAgent

__all__ = ["RiskAgent", "SentimentAgent"]
