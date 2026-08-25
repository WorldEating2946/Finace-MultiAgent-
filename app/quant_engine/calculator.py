"""
app/quant_engine/calculator.py — 财务计算引擎

本模块提供纯数学/统计计算能力，所有方法均为确定性的硬计算。
严禁在此模块中调用 LLM 或进行自然语言推理。

计算分类:
    - 同比/环比增长: calculate_yoy_growth / calculate_qoq_growth
    - 杜邦分析: calculate_dupont_analysis
    - 财务比率: calculate_financial_ratios (待扩展)
    - 估值模型: calculate_dcf / calculate_ddm (待扩展)

Author: 工藤
Date: 2026-08-05
Version: 0.1.0
"""

import logging

import numpy as np

from app.core.schemas import (
    DuPontAnalysisInput,
    DuPontAnalysisOutput,
    DuPontComponent,
    YoYBatchInput,
    YoYBatchOutput,
    YoYGrowthInput,
    YoYGrowthOutput,
)

logger = logging.getLogger(__name__)


# ============================================================================
# FinancialCalculator — 财务指标硬计算器
# ============================================================================


class FinancialCalculator:
    """财务指标硬计算器

    所有方法均为纯计算逻辑，不涉及网络 I/O、LLM 调用或数据库访问。
    输入严格使用 Pydantic 模型校验，输出同样为结构化 Pydantic 模型。

    使用示例:
        calc = FinancialCalculator()
        result = calc.calculate_yoy_growth(
            YoYGrowthInput(current_value=1000.0, previous_value=800.0, metric_name="营收")
        )
        print(f"同比增速: {result.growth_rate_pct:.2f}%")  # 25.00%
    """

    # ------------------------------------------------------------------
    # 同比增速计算 (YoY Growth)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_yoy_growth(input_data: YoYGrowthInput) -> YoYGrowthOutput:
        """计算单个指标的同比增速。

        公式:
            同比增速 = (本期数值 - 上年同期数值) / |上年同期数值|

        参数:
            input_data: YoYGrowthInput — 包含 current_value, previous_value, metric_name

        返回:
            YoYGrowthOutput — 包含增长率、绝对变动额、趋势判断

        异常:
            ValueError: 当 previous_value == 0 时抛出（除数不可为零）

        示例:
            >>> inp = YoYGrowthInput(current_value=1200, previous_value=1000, metric_name="营收")
            >>> FinancialCalculator.calculate_yoy_growth(inp)
            YoYGrowthOutput(metric_name="营收", growth_rate_pct=20.0, trend="上升")
        """
        current = input_data.current_value
        previous = input_data.previous_value

        # 防止除零 —— 财务上允许从 0 开始增长，此时设为 None 语义更准确
        if previous == 0.0:
            logger.warning(
                "指标 '%s': 上年同期值为 0，无法计算同比增速（除零）", input_data.metric_name
            )
            absolute_change = current - previous
            return YoYGrowthOutput(
                metric_name=input_data.metric_name,
                current_value=current,
                previous_value=previous,
                absolute_change=absolute_change,
                growth_rate=float("inf") if current > 0 else (float("-inf") if current < 0 else 0.0),
                growth_rate_pct=float("inf") if current > 0 else (float("-inf") if current < 0 else 0.0),
                trend="上升" if current > previous else ("下降" if current < previous else "持平"),
                period=input_data.period,
            )

        # 核心计算
        absolute_change = current - previous
        growth_rate = absolute_change / abs(previous)  # 小数形式
        growth_rate_pct = growth_rate * 100.0           # 百分比形式

        # 趋势判定
        if growth_rate > 0.001:      # 容差 0.1%
            trend = "上升"
        elif growth_rate < -0.001:
            trend = "下降"
        else:
            trend = "持平"

        logger.debug(
            "指标 '%s': 同比增速 = %.2f%% (%.2f → %.2f)",
            input_data.metric_name, growth_rate_pct, previous, current,
        )

        return YoYGrowthOutput(
            metric_name=input_data.metric_name,
            current_value=current,
            previous_value=previous,
            absolute_change=absolute_change,
            growth_rate=growth_rate,
            growth_rate_pct=growth_rate_pct,
            trend=trend,
            period=input_data.period,
        )

    @classmethod
    def calculate_yoy_batch(cls, batch_input: YoYBatchInput) -> YoYBatchOutput:
        """批量计算多个指标的同比增速。

        参数:
            batch_input: YoYBatchInput — 包含多个 YoYGrowthInput

        返回:
            YoYBatchOutput — 包含全部计算结果 + 汇总描述
        """
        results = [cls.calculate_yoy_growth(item) for item in batch_input.items]

        # 筛选有效增速（排除除零导致的 inf）
        valid_rates = [
            r.growth_rate_pct
            for r in results
            if abs(r.growth_rate_pct) != float("inf")
        ]

        # 生成汇总
        if not valid_rates:
            summary = "所有指标均无法计算有效同比增速（上年同期值均为 0）"
        else:
            avg_growth = np.mean(valid_rates)
            rising = sum(1 for r in results if r.trend == "上升")
            falling = sum(1 for r in results if r.trend == "下降")
            flat = sum(1 for r in results if r.trend == "持平")
            summary = (
                f"共计算 {len(results)} 项指标：{rising} 项上升、{falling} 项下降、"
                f"{flat} 项持平；平均增速 {avg_growth:+.2f}%"
            )

        return YoYBatchOutput(results=results, summary=summary)

    # ------------------------------------------------------------------
    # 杜邦分析 (DuPont Analysis)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_dupont_analysis(input_data: DuPontAnalysisInput) -> DuPontAnalysisOutput:
        """执行杜邦三因子分析。

        杜邦公式:
            ROE = 净利润率 × 资产周转率 × 权益乘数

        其中:
            净利润率 (Net Profit Margin) = 净利润 / 营业收入
            资产周转率 (Asset Turnover)    = 营业收入 / 总资产
            权益乘数 (Equity Multiplier)   = 总资产 / 股东权益

        参数:
            input_data: DuPontAnalysisInput

        返回:
            DuPontAnalysisOutput — 包含三因子分解、ROE 及验证

        使用示例:
            >>> inp = DuPontAnalysisInput(
            ...     net_income=100_000_000,
            ...     revenue=1_000_000_000,
            ...     total_assets=2_000_000_000,
            ...     shareholders_equity=800_000_000,
            ...     company_name="示例公司",
            ... )
            >>> FinancialCalculator.calculate_dupont_analysis(inp)
            DuPontAnalysisOutput(roe_pct=12.5, ...)
        """
        # 三因子计算
        net_profit_margin = input_data.net_income / input_data.revenue
        asset_turnover = input_data.revenue / input_data.total_assets
        equity_multiplier = input_data.total_assets / input_data.shareholders_equity

        components = DuPontComponent(
            net_profit_margin=round(net_profit_margin, 6),
            asset_turnover=round(asset_turnover, 6),
            equity_multiplier=round(equity_multiplier, 4),
        )

        # ROE = 三因子乘积
        roe = net_profit_margin * asset_turnover * equity_multiplier
        roe_pct = roe * 100.0

        # 交叉验证：ROE 也应等于 净利润 / 股东权益
        roe_direct = input_data.net_income / input_data.shareholders_equity
        roe_check = round(roe_direct, 6)

        # 若三因子乘积与直接计算结果存在显著偏差，记录警告
        if abs(roe - roe_direct) > 1e-9:
            logger.warning(
                "杜邦分析验证偏差: 三因子乘积=%.10f, 直接计算=%.10f, 偏差=%.2e",
                roe, roe_direct, abs(roe - roe_direct),
            )

        # 基础解读（不含 LLM，仅基于规则）
        interpretation = FinancialCalculator._interpret_dupont(
            net_profit_margin, asset_turnover, equity_multiplier
        )

        logger.info(
            "公司 '%s' (%s) 杜邦分析: ROE=%.2f%% (利润率=%.2f%%, 周转率=%.4f, 杠杆=%.2f)",
            input_data.company_name or "未知",
            input_data.period or "未知周期",
            roe_pct,
            net_profit_margin * 100,
            asset_turnover,
            equity_multiplier,
        )

        return DuPontAnalysisOutput(
            company_name=input_data.company_name,
            period=input_data.period,
            components=components,
            roe=round(roe, 6),
            roe_pct=round(roe_pct, 2),
            roe_check=roe_check,
            interpretation=interpretation,
        )

    @staticmethod
    def _interpret_dupont(
        net_profit_margin: float,
        asset_turnover: float,
        equity_multiplier: float,
    ) -> str:
        """基于规则阈值生成杜邦分析基础解读。

        此为纯规则判断，不依赖 LLM。更详细的商业解读应由 Report Agent 完成。
        """
        parts = []

        # 净利润率判断
        npm_pct = net_profit_margin * 100
        if npm_pct > 20:
            parts.append(f"净利润率较高（{npm_pct:.1f}%），产品或服务具备强定价权")
        elif npm_pct > 5:
            parts.append(f"净利润率处于合理水平（{npm_pct:.1f}%）")
        else:
            parts.append(f"净利润率偏低（{npm_pct:.1f}%），需关注成本控制")

        # 资产周转率判断
        if asset_turnover > 1.0:
            parts.append(f"资产周转率较高（{asset_turnover:.2f}），资产运营效率良好")
        elif asset_turnover > 0.3:
            parts.append(f"资产周转率适中（{asset_turnover:.2f}）")
        else:
            parts.append(f"资产周转率较低（{asset_turnover:.2f}），可能为重资产运营模式")

        # 权益乘数判断
        if equity_multiplier > 4.0:
            parts.append(f"权益乘数较高（{equity_multiplier:.2f}），财务杠杆水平偏高")
        elif equity_multiplier > 1.5:
            parts.append(f"权益乘数适中（{equity_multiplier:.2f}），资本结构相对均衡")
        else:
            parts.append(f"权益乘数较低（{equity_multiplier:.2f}），财务杠杆保守")

        return "；".join(parts)

    # ------------------------------------------------------------------
    # 辅助工具
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cagr(
        start_value: float,
        end_value: float,
        years: int,
    ) -> float:
        """计算年化复合增长率 (CAGR)。

        公式:
            CAGR = (终值 / 初值)^(1/年数) - 1

        参数:
            start_value: 期初值
            end_value: 期末值
            years: 年数（必须 >= 1）

        返回:
            float: CAGR（小数形式）

        异常:
            ValueError: years < 1 或 start_value <= 0
        """
        if years < 1:
            raise ValueError(f"年数必须 >= 1，当前值: {years}")
        if start_value <= 0:
            raise ValueError(f"期初值必须为正数，当前值: {start_value}")

        return (end_value / start_value) ** (1.0 / years) - 1.0

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """安全除法 —— 避免除零异常。

        参数:
            numerator: 分子
            denominator: 分母
            default: 分母为 0 时的默认返回值
        """
        return numerator / denominator if denominator != 0.0 else default
