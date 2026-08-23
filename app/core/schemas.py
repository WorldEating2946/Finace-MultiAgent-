"""
app/core/schemas.py — Pydantic V2 数据模型定义

本模块定义 FinanceAgent 全链路使用的结构化数据模型。
涵盖：财务计算输入/输出、市场数据请求/响应、工作流状态等。

Author: 工藤
Date: 2026-08-05
Version: 0.1.0
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
)

# ============================================================================
# 1. 枚举定义 — 统一管理项目中的常量枚举
# ============================================================================


class FinancialPeriod(str, Enum):
    """财务报表周期"""
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    H1 = "H1"       # 半年报
    FY = "FY"       # 年报


class MetricType(str, Enum):
    """财务指标类型"""
    REVENUE = "revenue"
    NET_PROFIT = "net_profit"
    TOTAL_ASSETS = "total_assets"
    TOTAL_LIABILITIES = "total_liabilities"
    SHAREHOLDERS_EQUITY = "shareholders_equity"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    GROSS_PROFIT = "gross_profit"
    EBIT = "ebit"
    EBITDA = "ebitda"


class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# 2. 基础数据容器 — 单条财务记录
# ============================================================================


class FinancialMetric(BaseModel):
    """单条财务指标数据点"""
    metric_type: MetricType = Field(..., description="指标类型")
    value: float = Field(..., description="指标数值（单位：元）")
    period: FinancialPeriod = Field(..., description="报表周期")
    fiscal_year: PositiveInt = Field(..., description="财年")
    report_date: date | None = Field(default=None, description="报表发布日期")
    currency: str = Field(default="CNY", description="货币单位")

    model_config = ConfigDict(use_enum_values=True)


class CompanyBasicInfo(BaseModel):
    """公司基本信息"""
    name: str = Field(..., min_length=1, max_length=128, description="公司全称")
    ticker: str | None = Field(default=None, max_length=16, description="股票代码")
    industry: str | None = Field(default=None, description="所属行业")
    listing_date: date | None = Field(default=None, description="上市日期")
    description: str | None = Field(default=None, description="公司简介")


# ============================================================================
# 3. 同比分析 (YoY Growth) — 输入/输出模型
# ============================================================================


class YoYGrowthInput(BaseModel):
    """同比增速计算输入"""
    current_value: float = Field(..., description="本期数值")
    previous_value: float = Field(..., description="上年同期数值")
    metric_name: str = Field(default="未命名指标", description="指标名称，用于结果标识")
    period: str | None = Field(default=None, description="当前周期标识，如 '2025Q4'")

    @field_validator("previous_value")
    @classmethod
    def previous_value_must_not_be_zero(cls, v: float) -> float:
        """技术上允许为 0，但应警告（会导致无穷大增速）"""
        return v

    model_config = ConfigDict(extra="forbid")


class YoYGrowthOutput(BaseModel):
    """同比增速计算输出"""
    metric_name: str = Field(..., description="指标名称")
    current_value: float = Field(..., description="本期数值")
    previous_value: float = Field(..., description="上年同期数值")
    absolute_change: float = Field(..., description="绝对变动额")
    growth_rate: float = Field(..., description="同比增速（小数形式，如 0.15 表示 15%）")
    growth_rate_pct: float = Field(..., description="同比增速（百分比形式，如 15.0 表示 15%）")
    trend: Literal["上升", "下降", "持平"] = Field(..., description="变动趋势")
    period: str | None = Field(default=None, description="周期标识")
    computed_at: datetime = Field(default_factory=datetime.now, description="计算时间戳")

    model_config = ConfigDict(extra="forbid")


class YoYBatchInput(BaseModel):
    """批量同比分析输入"""
    items: list[YoYGrowthInput] = Field(..., min_length=1, max_length=100, description="待计算项列表")


class YoYBatchOutput(BaseModel):
    """批量同比分析输出"""
    results: list[YoYGrowthOutput] = Field(..., description="计算结果列表")
    summary: str = Field(default="", description="汇总描述")


# ============================================================================
# 4. 杜邦分析 (DuPont Analysis) — 输入/输出模型
# ============================================================================


class DuPontAnalysisInput(BaseModel):
    """杜邦分析输入

    杜邦公式: ROE = 净利润率 × 资产周转率 × 权益乘数

    其中:
      - 净利润率 = 净利润 / 营业收入
      - 资产周转率 = 营业收入 / 总资产
      - 权益乘数 = 总资产 / 股东权益
    """
    net_income: float = Field(..., description="净利润")
    revenue: float = Field(..., description="营业收入")
    total_assets: float = Field(..., description="总资产")
    shareholders_equity: float = Field(..., description="股东权益（不含少数股东权益）")
    company_name: str | None = Field(default=None, description="公司名称")
    period: str | None = Field(default=None, description="分析周期，如 '2025FY'")

    @field_validator("revenue", "total_assets", "shareholders_equity")
    @classmethod
    def value_must_be_positive(cls, v: float) -> float:
        """分母项必须为正数"""
        if v <= 0:
            raise ValueError(f"分母项必须为正数，当前值: {v}")
        return v

    model_config = ConfigDict(extra="forbid")


class DuPontComponent(BaseModel):
    """杜邦分析三因子"""
    net_profit_margin: float = Field(..., description="净利润率 = 净利润 / 营业收入")
    asset_turnover: float = Field(..., description="资产周转率 = 营业收入 / 总资产")
    equity_multiplier: float = Field(..., description="权益乘数 = 总资产 / 股东权益")


class DuPontAnalysisOutput(BaseModel):
    """杜邦分析输出"""
    company_name: str | None = Field(default=None, description="公司名称")
    period: str | None = Field(default=None, description="分析周期")
    components: DuPontComponent = Field(..., description="三因子分解")
    roe: float = Field(..., description="净资产收益率 ROE（小数形式）")
    roe_pct: float = Field(..., description="净资产收益率（百分比形式）")
    roe_check: float = Field(..., description="验证值 = 三因子乘积，应与 roe 一致")
    interpretation: str = Field(default="", description="LLM 或规则生成的解读文本")
    computed_at: datetime = Field(default_factory=datetime.now, description="计算时间戳")

    model_config = ConfigDict(extra="forbid")


# ============================================================================
# 5. 市场数据服务 — 请求/响应模型
# ============================================================================


class MarketDataRequest(BaseModel):
    """市场数据获取请求"""
    ticker: str = Field(..., min_length=1, max_length=16, description="股票代码，如 '600519'")
    company_name: str | None = Field(default=None, description="公司名称（可选，用于校验）")
    start_date: date = Field(..., description="数据起始日期")
    end_date: date = Field(..., description="数据截止日期")
    metrics: list[MetricType] = Field(
        default_factory=lambda: [MetricType.REVENUE, MetricType.NET_PROFIT],
        description="需要获取的指标列表",
    )
    source: Literal["akshare", "tushare"] = Field(default="akshare", description="数据源")

    @field_validator("end_date")
    @classmethod
    def end_must_be_after_start(cls, v: date, info) -> date:
        """截止日期必须在起始日期之后"""
        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError(f"截止日期 {v} 不能早于起始日期 {start}")
        return v

    model_config = ConfigDict(extra="forbid")


class MarketDataResponse(BaseModel):
    """市场数据获取响应"""
    request: MarketDataRequest = Field(..., description="原始请求（回显）")
    data: list[FinancialMetric] = Field(default_factory=list, description="获取到的财务指标列表")
    source: str = Field(..., description="实际数据来源")
    fetched_at: datetime = Field(default_factory=datetime.now, description="数据获取时间")
    is_cached: bool = Field(default=False, description="是否来自缓存")
    error_msg: str | None = Field(default=None, description="错误信息（如有）")


# ============================================================================
# 6. 通用响应包装
# ============================================================================


class ServiceResult(BaseModel):
    """通用服务层返回包装

    用于统一 services 层的返回格式，使调用方无需关心具体异常处理。
    """
    success: bool = Field(..., description="操作是否成功")
    data: Any = Field(default=None, description="返回数据体")
    error_code: str | None = Field(default=None, description="错误码")
    error_msg: str | None = Field(default=None, description="错误描述")
    elapsed_ms: float | None = Field(default=None, description="耗时（毫秒）")

    model_config = ConfigDict(extra="forbid")


class PaginatedResult(ServiceResult):
    """分页结果包装"""
    total: int = Field(default=0, ge=0, description="总记录数")
    page: int = Field(default=1, ge=1, description="当前页码")
    page_size: int = Field(default=20, ge=1, le=200, description="每页条数")
