"""
app/agents/financial_agent/prompts.py — Financial Agent Prompt 模板

本模块管理 Financial Agent 与 LLM 交互的全部 Prompt。
包括 System Prompt（角色设定）和 User Prompt 构建函数。

设计原则:
    - System Prompt 设定角色边界，禁止模型编造数据
    - User Prompt 由硬计算结果填充，确保数值源头可追溯
    - 所有 Prompt 为中文，符合金融研报场景

Author: 工藤
Date: 2026-08-05
Version: 0.1.0
"""

# ============================================================================
# System Prompt — 资深 CFO 角色设定
# ============================================================================

FINANCIAL_AGENT_SYSTEM_PROMPT = """\
# 角色设定

你是一位拥有 20 年以上经验的资深 CFO（首席财务官），曾任职于多家上市公司。
你擅长从财务报表中快速识别企业的经营质量、盈利能力和潜在财务风险。

# 核心原则（必须遵守）

1. **数据驱动，严禁编造**：
   - 你的分析必须 100% 基于下方提供的硬计算数据。
   - 如果某个数据缺失或为 "N/A"，请直接说明"该指标数据暂缺"，严禁自行补全或猜测。
   - 严禁使用"大概""可能""估计"等模糊措辞来描述已提供的具体数值。

2. **专业客观**：
   - 保持中立立场，不夸大优势，不隐瞒风险。
   - 好就是好，差就是差，给出直白、专业的判断。
   - 避免使用营销话术或过度乐观的形容词。

3. **结构化表达**：
   - 先总后分：开头用 1-2 句话给出整体判断。
   - 然后按"盈利能力 → 成长性 → 资产效率 → 财务杠杆 → 综合风险提示"的顺序展开。
   - 每个维度提及对应的关键指标数值。

4. **风险意识**：
   - 如果 ROE < 0（亏损），必须明确指出经营困境。
   - 如果权益乘数 > 5（高杠杆），必须提示债务风险。
   - 如果营收增速持续下滑，必须警示成长性风险。

# 输出格式

请按以下结构输出分析（Markdown 格式）：

```
## 财务健康度总评
[1-2 句话整体判断]

## 盈利能力分析
[基于 ROE、净利润率等指标分析]

## 成长性分析
[基于营收同比、净利润同比分析趋势]

## 资产运营效率
[基于资产周转率分析]

## 财务杠杆与偿债风险
[基于权益乘数分析]

## 综合风险提示
[列出 2-3 个最值得关注的风险点或亮点]
```
"""


# ============================================================================
# User Prompt 构建函数
# ============================================================================


def build_analysis_prompt(
    company_name: str,
    ticker: str,
    period: str,
    key_metrics: dict,
    dupont: dict,
    yoy_history: list[dict],
) -> str:
    """根据硬计算结果构建发送给 LLM 的分析 Prompt。

    所有数值由调用方从 FinancialCalculator 输出中提取并传入，
    确保 LLM 不参与任何数值计算。

    参数:
        company_name: 公司全称
        ticker: 股票代码
        period: 分析周期描述，如 "2020-2024FY"
        key_metrics: 核心指标 dict
            {"roe_pct": 17.95, "net_profit_margin_pct": 9.03,
             "revenue_yoy_pct": 15.13, "net_profit_yoy_pct": 43.64,
             "equity_multiplier": 2.5, "asset_turnover": 0.8}
        dupont: 杜邦分析 dict
            {"net_profit_margin": 0.0903, "asset_turnover": 0.7955,
             "equity_multiplier": 2.5, "roe_computed": 0.1795,
             "roe_direct": 0.1795}
        yoy_history: 历年同比数据 list[dict]
            [{"period": "2024 vs 2023",
              "revenue_growth_pct": 15.13, "revenue_trend": "上升",
              "net_profit_growth_pct": 43.64, "profit_trend": "上升"}, ...]

    返回:
        str: 组装好的 User Prompt
    """
    # ── 核心指标摘要 ──────────────────────────────────────
    metrics_lines = [
        f"- 净资产收益率 (ROE): {_fmt_pct(key_metrics.get('roe_pct'))}",
        f"- 净利润率: {_fmt_pct(key_metrics.get('net_profit_margin_pct'))}",
        f"- 营收同比增速: {_fmt_pct(key_metrics.get('revenue_yoy_pct'))}",
        f"- 净利润同比增速: {_fmt_pct(key_metrics.get('net_profit_yoy_pct'))}",
        f"- 权益乘数: {key_metrics.get('equity_multiplier', 'N/A'):.2f}" if isinstance(key_metrics.get('equity_multiplier'), (int, float)) else f"- 权益乘数: {key_metrics.get('equity_multiplier', 'N/A')}",
        f"- 资产周转率: {key_metrics.get('asset_turnover', 'N/A'):.4f}" if isinstance(key_metrics.get('asset_turnover'), (int, float)) else f"- 资产周转率: {key_metrics.get('asset_turnover', 'N/A')}",
    ]

    # ── 杜邦拆解 ──────────────────────────────────────────
    dupont_lines = [
        f"- 净利润率 = 净利润 / 营业收入 = {dupont.get('net_profit_margin', 0):.4f} ({dupont.get('net_profit_margin', 0) * 100:.2f}%)",
        f"- 资产周转率 = 营业收入 / 总资产 = {dupont.get('asset_turnover', 0):.4f}",
        f"- 权益乘数 = 总资产 / 股东权益 = {dupont.get('equity_multiplier', 0):.2f}",
        f"- ROE = 三因子乘积 = {dupont.get('roe_computed', 0) * 100:.2f}%",
        f"- ROE 交叉验证（净利润/股东权益）= {dupont.get('roe_direct', 0) * 100:.2f}%",
    ]

    # ── 历年增速 ──────────────────────────────────────────
    history_lines = []
    for item in yoy_history:
        rev = _fmt_pct(item.get("revenue_growth_pct"))
        prf = _fmt_pct(item.get("net_profit_growth_pct"))
        rev_trend = item.get("revenue_trend", "N/A")
        prf_trend = item.get("profit_trend", "N/A")
        history_lines.append(
            f"- {item.get('period', 'N/A')}: "
            f"营收 {rev}（{rev_trend}），"
            f"净利润 {prf}（{prf_trend}）"
        )

    # ── 组装 ──────────────────────────────────────────────
    prompt = f"""\
请基于以下硬计算数据，对 {company_name}（{ticker}）的财务健康状况进行专业分析。

## 分析期间
{period}

## 核心财务指标
{chr(10).join(metrics_lines)}

## 杜邦分析三因子拆解
{chr(10).join(dupont_lines)}

## 历年同比增速趋势
{chr(10).join(history_lines) if history_lines else '（无历年数据）'}

---
请以上述数据为依据，按照你作为资深CFO的专业判断给出分析。
记住：绝不编造任何未在上述数据中出现的数字。"""
    return prompt


# ============================================================================
# 辅助函数
# ============================================================================


def _fmt_pct(value) -> str:
    """安全格式化百分比值"""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if abs(value) == float("inf"):
            return "N/A（上年同期为0）"
        return f"{value:+.2f}%"
    return str(value)
