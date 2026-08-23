"""
app/tools — 外部能力封装与通用工具

本层包含:
    - 外部能力封装：新闻抓取 / 情感评分 / 热点聚类 / 风险评估（LangChain @tool）
    - 轻量级通用工具函数：日期转换 / 货币换算 / 字符串清洗 / 数值精度

所有"重逻辑"和"外部交互"请放到 app/services。
"""

from .news_tools import fetch_recent_news, get_news_tools
from .risk_tools import (
    RISK_DIMENSIONS,
    assess_financial_risk,
    assess_industry_risk,
    assess_sentiment_risk,
    get_risk_tools,
    synthesize_risk,
)
from .sentiment_tools import (
    batch_score_news,
    cluster_topics,
    get_sentiment_tools,
    score_sentiment,
)

__all__ = [
    "RISK_DIMENSIONS",
    "assess_financial_risk",
    "assess_industry_risk",
    "assess_sentiment_risk",
    "batch_score_news",
    "cluster_topics",
    "fetch_recent_news",
    "get_news_tools",
    "get_risk_tools",
    "get_sentiment_tools",
    "score_sentiment",
    "synthesize_risk",
]
