"""
API 层的 Request / Response Schema

和 app/models 里的业务 Schema 分开：
  - models/*  = 内部业务模型（Agent 之间传递）
  - api/schemas = 对外 API 契约（前端 ↔ 后端）
"""

from pydantic import BaseModel, Field

# ════════════════════════════════════════════════════════════
# 请求
# ════════════════════════════════════════════════════════════

class SentimentRequest(BaseModel):
    """舆情分析请求"""
    symbol:       str = Field(..., description="股票代码", examples=["300750"])
    company_name: str = Field(..., description="企业名称", examples=["宁德时代"])
    days:         int = Field(default=30, ge=1, le=365, description="新闻回溯天数")


class RiskRequest(BaseModel):
    """
    风险评估请求。

    有两种用法：
    1. 只传 sentiment_result → 跳过 Phase 1，直接评估（前端已经拿到舆情数据）
    2. 同时传 sentiment_request → 先跑舆情再跑风险（一站式）
    """
    symbol:       str   = Field(..., description="股票代码")
    company_name: str   = Field(..., description="企业名称")
    days:         int   = Field(default=30, ge=1, le=365, description="新闻回溯天数")

    # 财务数据（可选——不传就用空数据）
    revenue_growth:    float | None = Field(default=None, description="营收增长率，如 0.15")
    gross_margin:      float | None = Field(default=None, description="毛利率")
    net_profit_margin: float | None = Field(default=None, description="净利率")
    debt_ratio:        float | None = Field(default=None, description="资产负债率")
    free_cash_flow:    float | None = Field(default=None, description="自由现金流(亿元)")
    anomalies:         list[str]    = Field(default_factory=list, description="财务异常项")


# ════════════════════════════════════════════════════════════
# 响应
# ════════════════════════════════════════════════════════════

class APIResponse(BaseModel):
    """统一 API 响应格式"""
    success: bool   = Field(..., description="请求是否成功")
    message: str    = Field(default="", description="提示信息")
    data:    dict   = Field(default_factory=dict, description="结构化数据")


class SentimentResponse(APIResponse):
    """舆情分析响应"""
    data: dict = Field(default_factory=lambda: {
        "symbol": "",
        "company_name": "",
        "searched_news_count": 0,
        "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
        "topics": [],
        "summary": "",
        "scored_news": [],
    })


class RiskResponse(APIResponse):
    """风险评估响应"""
    data: dict = Field(default_factory=lambda: {
        "symbol": "",
        "company_name": "",
        "overall_risk_level": "low",
        "overall_score": 0.0,
        "dimensions": [],
        "key_risks": [],
        "risk_summary": "",
        "reasoning_chain": "",
    })
