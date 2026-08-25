"""
app/agents/financial_agent/node.py — Financial Agent 核心节点

本模块是 Financial Agent 的执行入口，负责编排完整的财务分析链路:

    State 输入
       │
       ▼
    1. 提取参数 (ticker, company, date range)
       │
       ▼
    2. MarketDataService.fetch_financial_data()  ← 异步 I/O
       │
       ▼
    3. FinancialCalculator.calculate_yoy_growth()     ← CPU 硬计算
       FinancialCalculator.calculate_dupont_analysis()
       │
       ▼
    4. build_analysis_prompt() + LLM 调用            ← 异步 I/O
       │
       ▼
    5. FinancialAgentOutput → state.financial_result  ← 结构化输出

设计原则:
    - 全异步: I/O 操作使用 async/await，CPU 计算同步执行
    - 隔离: LLM 只做解读，数值 100% 来自 FinancialCalculator
    - 容错: API 失败 → 本地 JSON 文件 → 无数据降级；LLM 失败 → 规则引擎

Author: 工藤
Date: 2026-08-05
Version: 0.1.0
"""

import json
import logging
import os
import traceback
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from app.agents.financial_agent.prompts import (
    FINANCIAL_AGENT_SYSTEM_PROMPT,
    build_analysis_prompt,
)
from app.agents.financial_agent.schemas import (
    DuPontBreakdown,
    FinancialAgentInput,
    FinancialAgentOutput,
    KeyMetrics,
    YoYSummary,
)
from app.core.schemas import (
    DuPontAnalysisInput,
    MarketDataRequest,
    MetricType,
    YoYGrowthInput,
)
from app.quant_engine.calculator import FinancialCalculator
from app.services.data_fetcher import MarketDataService
from app.workflow.state import ResearchState

logger = logging.getLogger(__name__)


# ============================================================================
# 主入口 — 供 LangGraph graph.py 调用的节点函数
# ============================================================================


async def financial_analysis_node(state: ResearchState) -> dict[str, Any]:
    """Financial Agent 核心节点 — 完整的财务分析异步链路。

    此函数被 LangGraph 主图注册为 "financial" 节点。
    在 Send API 并行扇出后独立执行，与 research / sentiment 节点并发。

    参数:
        state: ResearchState — 全链路共享状态

    返回:
        dict: 部分状态更新，LangGraph 自动合并到全局 State
              {"financial_result": FinancialAgentOutput.model_dump()}

    异常处理:
        任何环节失败均会被捕获，错误信息写入返回 dict，不会中断整个 Workflow。
    """
    company = state.get("company", "未命名企业")
    ticker = state.get("ticker", "")
    logger.info("[FinancialAgent] === 开始财务分析: %s (ticker=%s) ===", company, ticker)

    try:
        # ── 1. 解析输入参数 ──────────────────────────────────
        agent_input = FinancialAgentInput(
            ticker=ticker or "000000",
            company_name=company,
            start_date=date(2020, 1, 1),
            end_date=date.today(),
            fiscal_years=5,
        )
        logger.debug("[FinancialAgent] 输入参数: %s", agent_input.model_dump_json())

        # ── 2. 异步获取原始财务数据 ──────────────────────────
        raw_metrics, fetch_error = await _fetch_financial_data(agent_input)

        # ── 3. 硬计算: 同比增速 + 杜邦分析 ────────────────────
        calc = FinancialCalculator()
        fy_data = _prepare_fiscal_data(raw_metrics, agent_input)

        # 3a. 无数据 → 提前返回，避免空 dict 导致后续计算崩溃
        if not fy_data:
            logger.warning("[FinancialAgent] 无可用财务数据，返回空结果")
            return {"financial_result": _build_no_data_output(company, ticker, fetch_error or "")}

        # 3b. 逐年同比计算
        yoy_summaries = _compute_yoy_growth(calc, fy_data, company)

        # 3c. 杜邦分析（最新年度）
        dupont_breakdown = _compute_dupont(calc, fy_data, company)

        # ── 4. 组装核心指标 ──────────────────────────────────
        latest_yoy = yoy_summaries[-1] if yoy_summaries else None
        key_metrics = KeyMetrics(
            roe_pct=dupont_breakdown.roe_computed * 100,
            net_profit_margin_pct=dupont_breakdown.net_profit_margin * 100,
            revenue_yoy_pct=latest_yoy.revenue_growth_pct if latest_yoy else None,
            net_profit_yoy_pct=latest_yoy.net_profit_growth_pct if latest_yoy else None,
            equity_multiplier=dupont_breakdown.equity_multiplier,
            asset_turnover=dupont_breakdown.asset_turnover,
        )

        # ── 5. 调用 LLM 生成财务点评 ─────────────────────────
        commentary = await _generate_commentary(
            agent_input=agent_input,
            key_metrics=key_metrics,
            dupont_breakdown=dupont_breakdown,
            yoy_summaries=yoy_summaries,
        )

        # ── 6. 组装最终输出 ──────────────────────────────────
        output = FinancialAgentOutput(
            company=company,
            ticker=ticker,
            analysis_period=f"{agent_input.start_date} ~ {agent_input.end_date}",
            key_metrics=key_metrics,
            dupont=dupont_breakdown,
            yoy_history=yoy_summaries,
            commentary=commentary,
            raw_calculations={
                "fy_data": fy_data,
            },
            data_source=_resolve_data_source(raw_metrics),
            fetch_error=fetch_error,
        )

        logger.info(
            "[FinancialAgent] === 分析完成: ROE=%.2f%%, 营收同比=%s, 点评长度=%d 字 ===",
            key_metrics.roe_pct,
            f"{key_metrics.revenue_yoy_pct:+.2f}%" if key_metrics.revenue_yoy_pct else "N/A",
            len(commentary),
        )

        return {"financial_result": output.model_dump()}

    except Exception:
        logger.exception("[FinancialAgent] 未捕获异常，触发顶层兜底")
        return _build_error_output(
            company=company,
            ticker=ticker or "N/A",
            error=traceback.format_exc(),
            period="N/A",
        )


# ============================================================================
# 内部辅助函数
# ============================================================================


def _build_error_output(
    company: str,
    ticker: str,
    error: str,
    period: str,
) -> dict[str, Any]:
    """构建异常情况下的降级输出。

    当 financial_analysis_node 任意环节抛出未捕获异常时被调用，
    生成一个最小有效的 FinancialAgentOutput dict，确保:
        - 下游 Risk Agent 不会因 financial_result 缺失而崩溃
        - 最终 Report 中能看到错误信息，便于排查
        - State.errors 由 Risk Agent 统一追加（本函数不直接写入 state）

    参数:
        company: 公司名称
        ticker: 股票代码
        error: 完整异常堆栈 (traceback.format_exc())
        period: 分析周期描述（异常时通常为 "N/A"）

    返回:
        dict: {"financial_result": FinancialAgentOutput.model_dump()}
    """
    logger.error(
        "[FinancialAgent] 生成错误降级输出: company=%s, error_summary=%s",
        company,
        error.split("\n")[-2] if "\n" in error else error[:100],
    )

    # 截取错误摘要用于报告展示（完整堆栈写入 fetch_error）
    error_lines = [ln.strip() for ln in error.strip().split("\n") if ln.strip()]
    error_summary = error_lines[-1] if error_lines else error[:200]

    output = FinancialAgentOutput(
        company=company,
        ticker=ticker,
        analysis_period=period,
        key_metrics=KeyMetrics(
            roe_pct=0.0,
            net_profit_margin_pct=0.0,
            equity_multiplier=0.0,
            asset_turnover=0.0,
        ),
        dupont=DuPontBreakdown(
            net_profit_margin=0.0,
            asset_turnover=0.0,
            equity_multiplier=0.0,
            roe_computed=0.0,
            roe_direct=0.0,
        ),
        commentary=(
            f"## ⚠️ 财务分析异常\n\n"
            f"财务分析节点发生未预期异常，所有指标暂不可用。\n\n"
            f"**错误摘要**: {error_summary}\n\n"
            f"请检查日志获取完整堆栈信息。"
        ),
        data_source="error",
        fetch_error=error,
    )
    return {"financial_result": output.model_dump()}


# ============================================================================
# 本地数据加载
# ============================================================================


# 默认数据目录（项目根目录下的 app/data/）
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
# 默认样本数据路径
_DEFAULT_FIXTURE_PATH = _DATA_DIR / "fixtures" / "sample_company.json"


def _load_local_fixture(
    file_path: str | None = None,
    ticker: str | None = None,
) -> list[dict[str, Any]] | None:
    """从本地 JSON/CSV 文件加载财务数据。

    查找优先级:
        1. file_path 参数显式指定的路径
        2. FINANCE_AGENT_DATA_FILE 环境变量
        3. ticker 自动匹配: app/data/{ticker}_*.json

    数据格式要求:
        {
            "fiscal_data": {
                "2024": {"revenue": ..., "net_profit": ..., "total_assets": ..., "shareholders_equity": ...},
                ...
            }
        }
    或数组格式:
        [{"fiscal_year": 2024, "metric_type": "revenue", "value": ...}, ...]

    参数:
        file_path: 显式文件路径（最高优先级）
        ticker: 股票代码，用于自动匹配 app/data/{ticker}_*.json

    返回:
        list[dict] 或 None
    """
    # 优先级 1: 显式路径
    resolved = file_path or os.getenv("FINANCE_AGENT_DATA_FILE", "")

    # 优先级 2: ticker 自动匹配 — 在 app/data/ 下搜索
    if not resolved and ticker:
        candidates = sorted(_DATA_DIR.glob(f"{ticker}_*.json"))
        if candidates:
            resolved = str(candidates[0])
            logger.info("[FinancialAgent] ticker=%s 自动匹配文件: %s", ticker, resolved)

    if not resolved:
        return None

    path = Path(resolved)
    if not path.is_file():
        logger.warning("[FinancialAgent] 本地数据文件不存在: %s", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # 格式 1: {"fiscal_data": {"2024": {...}, ...}}
        if isinstance(data, dict) and "fiscal_data" in data:
            raw: list[dict[str, Any]] = []
            for year_str, metrics in data["fiscal_data"].items():
                year = int(year_str)
                for metric_name, value in metrics.items():
                    raw.append({
                        "fiscal_year": year,
                        "metric_type": metric_name,
                        "value": float(value),
                    })
            logger.info("[FinancialAgent] 从本地文件加载 %d 条数据: %s", len(raw), path)
            return raw

        # 格式 2: 直接是数组
        if isinstance(data, list):
            logger.info("[FinancialAgent] 从本地文件加载 %d 条数据 (数组格式): %s", len(data), path)
            return data

        logger.warning("[FinancialAgent] 无法识别的本地数据格式: %s", path)
        return None

    except Exception as exc:
        logger.error("[FinancialAgent] 本地数据文件读取失败: %s, 错误: %s", path, exc)
        return None


# ============================================================================
# 内部辅助函数
# ============================================================================


def _build_no_data_output(company: str, ticker: str, reason: str) -> dict[str, Any]:
    """构建"无可用数据"的降级输出。

    与 _build_error_output 的区别:
        - 前者: 预期内的无数据（API 空 + 无本地文件）
        - 后者: 未预期的运行时异常
    """
    logger.info("[FinancialAgent] 生成无数据降级输出: company=%s, reason=%s", company, reason)

    output = FinancialAgentOutput(
        company=company,
        ticker=ticker,
        analysis_period="N/A",
        key_metrics=KeyMetrics(
            roe_pct=0.0,
            net_profit_margin_pct=0.0,
            equity_multiplier=0.0,
            asset_turnover=0.0,
        ),
        dupont=DuPontBreakdown(
            net_profit_margin=0.0,
            asset_turnover=0.0,
            equity_multiplier=0.0,
            roe_computed=0.0,
            roe_direct=0.0,
        ),
        commentary=(
            f"## ⚠️ 无可用财务数据\n\n"
            f"当前分析无法获取 **{company}** 的财务数据。\n\n"
            f"**原因**: {reason}\n\n"
            f"**建议**:\n"
            f"1. 确认股票代码 (ticker) 是否正确\n"
            f"2. 设置 `FINANCE_AGENT_DATA_FILE` 环境变量指向本地数据文件\n"
            f"3. 检查 AkShare/Tushare API 是否可正常访问"
        ),
        data_source="none",
        fetch_error=reason,
    )
    return {"financial_result": output.model_dump()}


def _resolve_data_source(raw_metrics: list[dict[str, Any]]) -> str:
    """解析实际数据来源标识。"""
    if not raw_metrics:
        return "none"
    return "api"


async def _fetch_financial_data(
    agent_input: FinancialAgentInput,
) -> tuple[list[dict[str, Any]], str | None]:
    """异步获取财务数据。

    尝试通过 MarketDataService 获取真实数据；
    失败时依次降级: 本地 JSON 文件 → 返回空列表。

    返回:
        (raw_metrics, error_msg)
    """
    ticker = agent_input.ticker
    if not ticker or ticker == "000000":
        logger.info("[FinancialAgent] 无有效 ticker，尝试从本地文件加载数据")
        local = _load_local_fixture(ticker=agent_input.ticker)
        if local:
            return local, None
        return [], "无有效 ticker 且无本地数据文件，设置 FINANCE_AGENT_DATA_FILE 环境变量可指定本地数据源"

    try:
        async with MarketDataService(timeout=30.0) as svc:
            request = MarketDataRequest(
                ticker=ticker,
                company_name=agent_input.company_name,
                start_date=agent_input.start_date,
                end_date=agent_input.end_date,
                metrics=[
                    MetricType.REVENUE,
                    MetricType.NET_PROFIT,
                    MetricType.TOTAL_ASSETS,
                    MetricType.SHAREHOLDERS_EQUITY,
                ],
                source="akshare",
            )
            response = await svc.fetch_financial_data(request)

            if response.error_msg:
                logger.warning("[FinancialAgent] API 错误: %s，尝试本地文件降级", response.error_msg)
                local = _load_local_fixture(ticker=agent_input.ticker)
                if local:
                    return local, None
                return [], response.error_msg

            if response.data:
                raw = [m.model_dump() for m in response.data]
                logger.info("[FinancialAgent] 成功获取 %d 条真实财务数据", len(raw))
                return raw, None

            logger.info("[FinancialAgent] API 返回空数据，尝试本地文件降级")
            local = _load_local_fixture(ticker=agent_input.ticker)
            if local:
                return local, None
            return [], "API 返回空数据且无本地数据文件"

    except Exception as exc:
        logger.warning("[FinancialAgent] MarketDataService 异常: %s，尝试本地文件降级", exc)
        local = _load_local_fixture(ticker=agent_input.ticker)
        if local:
            return local, None
        return [], str(exc)


def _prepare_fiscal_data(
    raw_metrics: list[dict[str, Any]],
    agent_input: FinancialAgentInput,
) -> dict[int, dict[str, float]]:
    """将原始指标列表整理为 {财年: {指标: 值}} 结构。

    若 raw_metrics 为空，返回空 dict；调用方负责处理无数据情况。
    """
    if raw_metrics:
        fy_data: dict[int, dict[str, float]] = {}
        for m in raw_metrics:
            year = m.get("fiscal_year")
            if year is None:
                continue
            if year not in fy_data:
                fy_data[year] = {
                    "revenue": 0.0,
                    "net_profit": 0.0,
                    "total_assets": 0.0,
                    "shareholders_equity": 0.0,
                }
            metric_type = m.get("metric_type", "")
            value = float(m.get("value", 0.0))
            if metric_type == "revenue":
                fy_data[year]["revenue"] = value
            elif metric_type == "net_profit":
                fy_data[year]["net_profit"] = value
            elif metric_type == "total_assets":
                fy_data[year]["total_assets"] = value
            elif metric_type == "shareholders_equity":
                fy_data[year]["shareholders_equity"] = value
        if fy_data:
            return fy_data

    # 无数据可用 — 返回空 dict，由调用方处理
    logger.warning("[FinancialAgent] 无财务数据可用: API 为空且本地文件均未加载")
    return {}


def _compute_yoy_growth(
    calc: FinancialCalculator,
    fy_data: dict[int, dict[str, float]],
    company: str,
) -> list[YoYSummary]:
    """逐年计算同比增速。

    对有序的财年数据逐对调用 FinancialCalculator.calculate_yoy_growth()。
    """
    summaries: list[YoYSummary] = []
    sorted_years = sorted(fy_data.keys())

    for i, year in enumerate(sorted_years):
        current = fy_data[year]
        if i == 0:
            # 基准年: 无上年对比
            summaries.append(YoYSummary(
                period=f"{year} (基准年)",
                revenue_growth_pct=None,
                net_profit_growth_pct=None,
                revenue_trend="持平",
                profit_trend="持平",
            ))
            continue

        prev = fy_data[sorted_years[i - 1]]

        # 营收同比
        rev_output = calc.calculate_yoy_growth(
            YoYGrowthInput(
                current_value=current["revenue"],
                previous_value=prev["revenue"],
                metric_name=f"{company} 营收",
                period=f"{year} vs {sorted_years[i - 1]}",
            )
        )

        # 净利润同比
        prf_output = calc.calculate_yoy_growth(
            YoYGrowthInput(
                current_value=current["net_profit"],
                previous_value=prev["net_profit"],
                metric_name=f"{company} 净利润",
                period=f"{year} vs {sorted_years[i - 1]}",
            )
        )

        summaries.append(YoYSummary(
            period=f"{year} vs {sorted_years[i - 1]}",
            revenue_growth_pct=(
                rev_output.growth_rate_pct
                if abs(rev_output.growth_rate_pct) != float("inf")
                else None
            ),
            net_profit_growth_pct=(
                prf_output.growth_rate_pct
                if abs(prf_output.growth_rate_pct) != float("inf")
                else None
            ),
            revenue_trend=rev_output.trend,
            profit_trend=prf_output.trend,
        ))

    return summaries


def _compute_dupont(
    calc: FinancialCalculator,
    fy_data: dict[int, dict[str, float]],
    company: str,
) -> DuPontBreakdown:
    """执行杜邦三因子分析（最新财年）。"""
    latest_year = max(fy_data.keys())
    latest = fy_data[latest_year]

    dupont_input = DuPontAnalysisInput(
        net_income=latest["net_profit"],
        revenue=latest["revenue"],
        total_assets=latest["total_assets"],
        shareholders_equity=latest["shareholders_equity"],
        company_name=company,
        period=f"{latest_year}FY",
    )
    dupont_output = calc.calculate_dupont_analysis(dupont_input)

    return DuPontBreakdown(
        net_profit_margin=dupont_output.components.net_profit_margin,
        asset_turnover=dupont_output.components.asset_turnover,
        equity_multiplier=dupont_output.components.equity_multiplier,
        roe_computed=dupont_output.roe,
        roe_direct=dupont_output.roe_check,
    )


# ============================================================================
# LLM 调用
# ============================================================================


async def _generate_commentary(
    agent_input: FinancialAgentInput,
    key_metrics: KeyMetrics,
    dupont_breakdown: DuPontBreakdown,
    yoy_summaries: list[YoYSummary],
) -> str:
    """调用 LLM 生成财务健康度点评。

    将硬计算结果组装为 Prompt → 调用 DeepSeek API → 返回点评文本。

    容错:
        - LLM 不可用时: 返回基于规则的简短点评
        - 超时时: 返回超时提示
    """
    # 组装 User Prompt
    user_prompt = build_analysis_prompt(
        company_name=agent_input.company_name,
        ticker=agent_input.ticker,
        period=f"{agent_input.start_date} ~ {agent_input.end_date}",
        key_metrics=key_metrics.model_dump(),
        dupont=dupont_breakdown.model_dump(),
        yoy_history=[s.model_dump() for s in yoy_summaries],
    )

    # 获取 LLM 配置（从环境变量，安全策略）
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("DEEPSEEK_MODEL_CHAT", "deepseek-v4-flash")

    if not api_key:
        logger.warning("[FinancialAgent] DEEPSEEK_API_KEY 未配置，使用规则降级")
        return _rule_based_commentary(key_metrics)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": FINANCIAL_AGENT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000,
                    "stream": False,
                },
            )

            if response.status_code != 200:
                logger.error(
                    "[FinancialAgent] LLM API 返回 %d: %s",
                    response.status_code,
                    response.text[:200],
                )
                return _rule_based_commentary(key_metrics)

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            logger.info(
                "[FinancialAgent] LLM 点评生成完成 (model=%s, tokens=%s)",
                data.get("model", "unknown"),
                data.get("usage", {}).get("total_tokens", "unknown"),
            )
            return content

    except httpx.TimeoutException:
        logger.error("[FinancialAgent] LLM 调用超时")
        return "（LLM 调用超时，以下为规则生成的简要分析）\n\n" + _rule_based_commentary(key_metrics)

    except Exception as exc:
        logger.error("[FinancialAgent] LLM 调用异常: %s", exc)
        return "（LLM 服务不可用，以下为规则生成的简要分析）\n\n" + _rule_based_commentary(key_metrics)


def _rule_based_commentary(metrics: KeyMetrics) -> str:
    """基于规则的降级点评 —— 当 LLM 不可用时的后备方案。

    纯规则判断，不含 LLM 调用，确保系统在任何情况下都有输出。
    """
    lines = ["## 财务健康度总评（规则引擎生成）", ""]

    # ROE 判断
    roe = metrics.roe_pct
    if roe > 20:
        lines.append(f"公司 ROE 为 {roe:.2f}%，处于优秀水平，股东回报能力强。")
    elif roe > 10:
        lines.append(f"公司 ROE 为 {roe:.2f}%，处于良好水平。")
    elif roe > 0:
        lines.append(f"公司 ROE 为 {roe:.2f}%，处于一般水平，盈利能力有待提升。")
    else:
        lines.append(f"⚠️ 公司 ROE 为 {roe:.2f}%，处于亏损状态，经营面临严重困难。")

    # 成长性
    rev_yoy = metrics.revenue_yoy_pct
    if rev_yoy is not None:
        if rev_yoy > 20:
            lines.append(f"营收同比增长 {rev_yoy:+.2f}%，增速强劲。")
        elif rev_yoy > 0:
            lines.append(f"营收同比增长 {rev_yoy:+.2f}%，保持正向增长。")
        else:
            lines.append(f"⚠️ 营收同比 {rev_yoy:+.2f}%，出现下滑，需关注市场竞争力。")

    # 杠杆
    em = metrics.equity_multiplier
    if em > 4:
        lines.append(f"⚠️ 权益乘数 {em:.2f}，财务杠杆偏高，债务风险值得关注。")
    elif em > 2:
        lines.append(f"权益乘数 {em:.2f}，杠杆水平适中。")
    else:
        lines.append(f"权益乘数 {em:.2f}，财务杠杆保守，经营风格稳健。")

    # 效率
    at = metrics.asset_turnover
    if at > 1:
        lines.append(f"资产周转率 {at:.4f}，资产运营效率优秀。")
    elif at > 0.5:
        lines.append(f"资产周转率 {at:.4f}，运营效率良好。")
    else:
        lines.append(f"资产周转率 {at:.4f}，资金周转效率偏低，可能为重资产行业特征。")

    return "\n\n".join(lines)
