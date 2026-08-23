"""
app/agents/financial_agent/schemas.py — Financial Agent 输入/输出模型

本模块定义 Financial Agent 专属的 Pydantic V2 数据结构。
与 app/core/schemas.py 的关系:
    - core/schemas: 通用基础模型（FinancialMetric, DuPontAnalysisOutput 等）
    - 本文件: Financial Agent 特有的 I/O 协议

Author: 工藤
Date: 2026-08-05
Version: 0.1.0
"""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ============================================================================
# Financial Agent 输入
# ============================================================================


class FinancialAgentInput(BaseModel):
    """Financial Agent 输入参数

    从 ResearchState 中提取，定义本 Agent 需要的最小输入集。
    """
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=16,
        description="股票代码，如 '300750'（宁德时代）",
    )
    company_name: str = Field(
        default="未命名企业",
        min_length=1,
        max_length=128,
        description="公司全称，用于报告标识",
    )
    start_date: date = Field(
        default_factory=lambda: date(2020, 1, 1),
        description="数据起始日期",
    )
    end_date: date = Field(
        default_factory=lambda: date.today(),
        description="数据截止日期",
    )
    fiscal_years: int = Field(
        default=5,
        ge=2,
        le=10,
        description="需要回溯的财年数",
    )

    @field_validator("end_date")
    @classmethod
    def end_must_be_after_start(cls, v: date, info) -> date:
        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError(f"截止日期 {v} 不能早于起始日期 {start}")
        return v

    model_config = ConfigDict(extra="forbid")


# ============================================================================
# Financial Agent 输出
# ============================================================================


class KeyMetrics(BaseModel):
    """核心财务指标摘要"""
    # 盈利指标
    roe_pct: float = Field(..., description="净资产收益率 (%)")
    net_profit_margin_pct: float = Field(..., description="净利润率 (%)")
    # 成长指标
    revenue_yoy_pct: float | None = Field(default=None, description="营收同比增速 (%)")
    net_profit_yoy_pct: float | None = Field(default=None, description="净利润同比增速 (%)")
    # 偿债指标
    equity_multiplier: float = Field(..., description="权益乘数")
    asset_turnover: float = Field(..., description="资产周转率")

    model_config = ConfigDict(extra="forbid")


class DuPontBreakdown(BaseModel):
    """杜邦分析三因子拆解"""
    net_profit_margin: float = Field(..., description="净利润率 = 净利润 / 营业收入")
    asset_turnover: float = Field(..., description="资产周转率 = 营业收入 / 总资产")
    equity_multiplier: float = Field(..., description="权益乘数 = 总资产 / 股东权益")
    roe_computed: float = Field(..., description="ROE = 三因子乘积")
    roe_direct: float = Field(..., description="ROE = 净利润 / 股东权益（交叉验证）")

    model_config = ConfigDict(extra="forbid")


class YoYSummary(BaseModel):
    """同比增速摘要"""
    period: str = Field(..., description="周期标识，如 '2024 vs 2023'")
    revenue_growth_pct: float | None = Field(default=None, description="营收同比增速 (%)")
    net_profit_growth_pct: float | None = Field(default=None, description="净利润同比增速 (%)")
    revenue_trend: str = Field(default="持平", description="营收趋势: 上升/下降/持平")
    profit_trend: str = Field(default="持平", description="利润趋势: 上升/下降/持平")


class FinancialAgentOutput(BaseModel):
    """Financial Agent 完整输出

    此模型会被序列化为 dict 写入 ResearchState.financial_result。
    """
    # 元信息
    company: str = Field(..., description="公司名称")
    ticker: str = Field(..., description="股票代码")
    analysis_period: str = Field(..., description="分析周期描述")

    # 核心指标
    key_metrics: KeyMetrics = Field(..., description="核心财务指标摘要")

    # 杜邦拆解
    dupont: DuPontBreakdown = Field(..., description="杜邦分析三因子")

    # 历年增速
    yoy_history: list[YoYSummary] = Field(default_factory=list, description="逐年同比增速")

    # LLM 生成的自然语言分析
    commentary: str = Field(
        default="",
        description="LLM 生成的财务健康度点评（资深CFO视角）",
    )

    # 原始计算数据（用于下游 Agent 溯源）
    raw_calculations: dict[str, Any] = Field(
        default_factory=dict,
        description="FinancialCalculator 原始输出（序列化后）",
    )

    # 数据溯源
    data_source: str = Field(default="unknown", description="数据来源: api | sample | cache")
    fetch_error: str | None = Field(default=None, description="数据获取错误信息")
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="生成时间戳 (ISO 8601)",
    )

    model_config = ConfigDict(extra="forbid")


# ============================================================================
# LLM 交互模型
# ============================================================================


class LLMAnalysisRequest(BaseModel):
    """发送给 LLM 的分析请求"""
    system_prompt: str = Field(..., description="System Prompt")
    user_prompt: str = Field(..., description="组装好的用户 Prompt（含计算数据）")
    model: str = Field(default="deepseek-v4-flash", description="模型 ID")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="生成温度")


class LLMAnalysisResponse(BaseModel):
    """LLM 返回的分析结果"""
    content: str = Field(..., description="LLM 生成的财务点评文本")
    model: str = Field(default="", description="实际使用的模型")
    tokens_used: int | None = Field(default=None, description="消耗的 token 数")
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
    )
