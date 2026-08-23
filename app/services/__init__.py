"""
app/services — 业务服务层

本层负责：
    - 外部 API 调用（金融数据、搜索、LLM）
    - 数据库读写编排
    - 跨模块业务流程组合

与 app/tools 的边界：
    - services: 重逻辑、外部交互（如获取5年财报）
    - tools: 轻量通用工具（如日期格式转换、货币单位换算）
"""

from app.services.data_fetcher import MarketDataService

__all__ = ["MarketDataService"]
