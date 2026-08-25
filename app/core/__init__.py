"""
app/core — 核心配置与基础定义

包含:
    - schemas: Pydantic V2 数据模型（Financial Agent 计算 I/O）
    - config: 系统配置与参数管理（pydantic-settings，.env 读取）
    - logging: 日志配置
    - exceptions: 自定义异常
    - llm_factory: LLM 统一工厂（Sentiment & Risk Agent）
    - retry: 三层兜底机制（自动重试 → 降级 → 兜底）
"""

from app.core.config import Settings, get_settings
from app.core.llm_factory import LLMFactory, get_llm, get_structured_llm
from app.core.retry import AgentFallbackHandler, with_retry
from app.core.schemas import (
    DuPontAnalysisInput,
    DuPontAnalysisOutput,
    MarketDataRequest,
    MarketDataResponse,
    ServiceResult,
    YoYGrowthInput,
    YoYGrowthOutput,
)

__all__ = [
    "AgentFallbackHandler",
    "DuPontAnalysisInput",
    "DuPontAnalysisOutput",
    "LLMFactory",
    "MarketDataRequest",
    "MarketDataResponse",
    "ServiceResult",
    "Settings",
    "YoYGrowthInput",
    "YoYGrowthOutput",
    "get_llm",
    "get_settings",
    "get_structured_llm",
    "with_retry",
]
