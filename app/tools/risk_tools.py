"""
多源风险推理工具

融合舆情信号、财务异常与行业周期三个维度，输出风险评分和推导链条。
供 Risk Agent 调用。

════════════════════ 评分模型 ════════════════════════
三维度加权综合：
  - 舆情负面信号（40%）：新闻负面占比 + 高危主题出现频率
  - 财务异常信号（35%）：营收/利润/负债异常 + 财务异常项
  - 行业周期风险（25%）：政策/监管/供应链/竞争/技术替代

Score → Level 映射：
  score ≥ 0.7  → HIGH（需要立即关注）
  score ≥ 0.4  → MEDIUM（建议持续监控）
  score < 0.4  → LOW（当前未发现显著风险）

════════════════════ 设计原则 ════════════════════════
1. 每个维度独立评分、独立举证——某个维度挂了，其他维度照常
2. 评分逻辑是纯 Python 计算（不做 LLM 推理），保证确定性
3. RISK_DIMENSIONS 表驱动——改权重/加维度只改配置表，不动代码
═══════════════════════════════════════════════════════════
"""

from langchain_core.tools import tool

from ..models.sentiment_risk_models import (
    FinancialSummary,
    RiskAssessment,
    RiskDimension,
    RiskLevel,
    SentimentResult,
)

# ════════════════════════════════════════════════════════════
# 风险维度配置表（表驱动——改权重/加维度只改这一张表）
# ════════════════════════════════════════════════════════════

RISK_DIMENSIONS: list[dict] = [
    {
        "key": "sentiment",              # 维度标识（唯一）
        "name": "舆情负面信号",           # 人类可读的名称
        "weight": 0.40,                  # 权重（三维度之和应为 1.0）
        "tool": "assess_sentiment_risk", # 对应的评估工具函数名
        "tool_kwargs": lambda sr, fs: {"sentiment_result": sr},
        # ↑ 传参工厂：lambda 接收 (sentiment_result, financial_summary)，
        #   返回传给该评估工具的参数 dict。
        #   舆情维度只需要舆情数据，不需要财务数据，所以只传 sr。
    },
    {
        "key": "financial",
        "name": "财务异常信号",
        "weight": 0.35,
        "tool": "assess_financial_risk",
        "tool_kwargs": lambda sr, fs: {"financial": fs},
        # 财务维度只需要财务数据，不需要舆情数据，只传 fs。
    },
    {
        "key": "industry",
        "name": "行业周期与外部环境",
        "weight": 0.25,
        "tool": "assess_industry_risk",
        "tool_kwargs": lambda sr, fs: {"sentiment_result": sr, "financial": fs},
        # 行业维度需要舆情主题（看政策/监管关键词）+ 财务趋势，两个都要。
    },
]
"""
扩展示例——要加第四个维度（如'ESG 风险'，权重 0.10）：
  1. 在 RISK_DIMENSIONS 加一条
  2. 写一个 assess_esg_risk @tool 函数
  3. 在 _DIM_TOOL_MAP 注册
  4. 其他三维度权重从 0.40/0.35/0.25 调整为 0.35/0.30/0.25（保持总和 1.0）
不需要改 synthesize_risk 的主逻辑。
"""


# ════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════

def _score_to_level(score: float) -> RiskLevel:
    """
    将连续风险评分映射为离散等级。

    阈值设计考量：
      - 0.4：保守策略——负面舆情占比 30% 左右或单项财务异常即入 MEDIUM，
        宁可多提醒不漏报
      - 0.7：只有多维度同时触发或单个维度得分极高才入 HIGH，
        避免"狼来了"效应
    """
    if score >= 0.7:
        return RiskLevel.HIGH
    if score >= 0.4:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


# ════════════════════════════════════════════════════════════
# 各维度评分工具（独立可测——每个函数只评估一个维度）
# ════════════════════════════════════════════════════════════

@tool
def assess_sentiment_risk(sentiment_result: SentimentResult) -> RiskDimension:
    """
    从舆情结果中提取风险信号。

    分析逻辑（纯规则，不调 LLM，保证可复现）：
      1. 计算负面新闻占比 = negative 条数 / 总条数
      2. 检查热点主题是否命中高危关键词（监管/制裁/诉讼/关税等）
      3. 评分 = min(负面比 × 1.5 + 高危主题数 × 0.05, 1.0)

    为什么负面比要 ×1.5？
      负面舆情比普通信号更紧急。例如 30% 负面新闻 → 评分 0.45，
      已经触发 MEDIUM。如果同时命中 2 个高危主题 → 0.45 + 0.10 = 0.55。

    Args:
        sentiment_result: Sentiment Agent 的完整输出

    Returns:
        RiskDimension: 舆情维度的风险评分和证据
    """
    # ── 计算负面占比 ──
    dist = sentiment_result.sentiment_distribution
    total = sum(dist.values()) or 1     # 防除零：空列表时 total=1
    negative_ratio = dist.get("negative", 0) / total

    # ── 证据收集 ──
    evidence: list[str] = []

    if negative_ratio > 0.5:
        evidence.append(f"负面舆情占比 {negative_ratio:.0%}，超过 50%，市场情绪显著悲观")
    elif negative_ratio > 0.3:
        evidence.append(f"负面舆情占比 {negative_ratio:.0%}，超过 30%，需密切关注")

    # 高危关键词——任何一个出现在主题标签中就累加风险
    high_risk_keywords = {"监管", "制裁", "诉讼", "关税", "调查", "处罚", "违约", "退市"}
    for topic in sentiment_result.topics:
        if any(kw in topic.label for kw in high_risk_keywords):
            evidence.append(f"高危主题「{topic.label}」涉及 {topic.news_count} 条新闻")

    # ── 评分 ──
    score = min(
        negative_ratio * 1.5                          # 负面占比加权
        + 0.05 * sum(                                  # 高危主题加分
            1 for t in sentiment_result.topics
            if any(kw in t.label for kw in high_risk_keywords)
        ),
        1.0,                                           # 上限 1.0
    )

    return RiskDimension(
        dimension="舆情负面信号",
        score=round(score, 2),
        evidence=evidence or ["未发现明显负面舆情聚集"],
        reasoning=f"负面率 {negative_ratio:.0%}，共 {len(sentiment_result.topics)} 个热点主题",
    )


@tool
def assess_financial_risk(financial: FinancialSummary) -> RiskDimension:
    """
    从财务数据中识别风险信号。

    检查四个维度，每项触发一个 risk_flag：
      1. 营收负增长 → 严重信号（risk_flags += 1）
         营收增长 < 5% → 需关注（记录 evidence，但不计分）
      2. 毛利率 < 15% → 偏低，记录 evidence
      3. 资产负债率 > 70% → 严重信号
      4. 财务异常项列表 → 每项都计分（这些是 Financial Agent 的 LLM
         识别出的非标准异常，如'应收周转天数突增'）

    评分 = risk_flags / total_flags（触发比例）
    当财务数据完全正常时返回 0.0。

    Args:
        financial: Financial Agent 输出的财务摘要

    Returns:
        RiskDimension: 财务维度的风险评分和证据
    """
    evidence: list[str] = []
    risk_flags = 0    # 触发预警的指标数
    total_flags = 0   # 可评估的指标总数

    # ── 营收增长率 ──
    if financial.revenue_growth is not None:
        total_flags += 1
        if financial.revenue_growth < 0:
            evidence.append(f"营收负增长 {financial.revenue_growth:.1%}")
            risk_flags += 1
        elif financial.revenue_growth < 0.05:
            evidence.append(f"营收增长乏力，仅 {financial.revenue_growth:.1%}")

    # ── 毛利率 ──
    if financial.gross_margin is not None:
        total_flags += 1
        if financial.gross_margin < 0.15:
            evidence.append(f"毛利率偏低 {financial.gross_margin:.1%}，低于 15% 健康线")

    # ── 资产负债率 ──
    if financial.debt_ratio is not None:
        total_flags += 1
        if financial.debt_ratio > 0.7:
            evidence.append(f"资产负债率偏高 {financial.debt_ratio:.1%}，超过 70% 警戒线")
            risk_flags += 1

    # ── 财务异常项（Financial Agent 的 LLM 识别） ──
    for anomaly in (financial.anomalies or []):
        evidence.append(anomaly)
        risk_flags += 1
        total_flags += 1

    # 防除零：所有指标都缺失时 score = 0
    score = risk_flags / max(total_flags, 1)

    return RiskDimension(
        dimension="财务异常信号",
        score=round(score, 2),
        evidence=evidence or ["财务指标无明显异常"],
        reasoning=f"{risk_flags}/{total_flags} 项指标触发风险预警",
    )


@tool
def assess_industry_risk(sentiment_result: SentimentResult,
                         financial: FinancialSummary) -> RiskDimension:
    """
    评估行业周期与外部环境风险。

    分析方式：扫描舆情热点主题，检查是否命中行业风险关键词。
    每个匹配的关键词 = 一个风险信号，评分按命中数线性累加。

    为什么不用财务数据做行业判断？
      财务趋势（如毛利下滑）已在 assess_financial_risk 中评估。
      行业周期风险关注的是外部因素——政策变化、供应链扰动、技术替代——
      这些在新闻里比在财报里更早出现。新闻是领先指标，财报是滞后指标。

    Args:
        sentiment_result: 舆情结果（主要数据源）
        financial:        财务摘要（当前版本做趋势交叉验证预留，暂未使用）

    Returns:
        RiskDimension: 行业周期维度的风险评分
    """
    evidence: list[str] = []

    # 行业风险关键词映射表：舆情主题中出现的词 → 风险描述
    industry_keywords = {
        "政策":   "政策变动风险",
        "监管":   "监管趋严风险",
        "供应链": "供应链扰动风险",
        "关税":   "海外关税风险",
        "竞争":   "竞争加剧风险",
        "替代":   "技术替代风险",
        "周期":   "行业周期下行风险",
    }

    matched: set[str] = set()      # 用 set 去重——同一个关键词多次命中只计一次
    for topic in sentiment_result.topics:
        for kw, desc in industry_keywords.items():
            if kw in topic.label:
                matched.add(desc)
                evidence.append(f"舆情热点「{topic.label}」指向 {desc}")

    # 评分：每个命中 +0.25，最多 1.0
    score = min(len(matched) * 0.25, 1.0)

    return RiskDimension(
        dimension="行业周期与外部环境",
        score=round(score, 2),
        evidence=evidence or ["未发现显著行业周期风险信号"],
        reasoning=f"匹配 {len(matched)} 项行业风险关键词",
    )


# ════════════════════════════════════════════════════════════
# 综合风险判定（由 RISK_DIMENSIONS 表驱动）
# ════════════════════════════════════════════════════════════

@tool
def synthesize_risk(symbol: str,
                    company_name: str,
                    sentiment_result: SentimentResult,
                    financial: FinancialSummary) -> RiskAssessment:
    """
    综合多源信号，输出最终风险判定。

    ══════════════════ 执行流程 ══════════════════
    1. 按 RISK_DIMENSIONS 配置表，逐维度调用评估工具
    2. 加权综合 = Σ(维度评分 × 权重)
    3. 映射为风险等级
    4. 汇总所有维度的 evidence 为 key_risks
    5. 构建完整推导链条（每步推理可追溯）

    为什么把评分逻辑放在 @tool 里而不是在 Agent 的 run() 里？
      → @tool 可以被 LLM Function Calling 发现和调用（Phase 2）。
        如果以后需要 LLM 自主决定评估哪些维度，工具已经就绪。
    ════════════════════════════════════════════════

    Args:
        symbol:           股票代码
        company_name:     企业名称
        sentiment_result: Sentiment Agent 的输出
        financial:        Financial Agent 的输出

    Returns:
        RiskAssessment: 完整风险评估结果
    """
    # ── 维度评估工具注册表 ──
    _DIM_TOOL_MAP = {
        "assess_sentiment_risk": assess_sentiment_risk,
        "assess_financial_risk": assess_financial_risk,
        "assess_industry_risk": assess_industry_risk,
    }

    # ── 1. 逐维度评估（由 RISK_DIMENSIONS 配置表驱动）──
    dimension_scores: list[RiskDimension] = []
    for dim_cfg in RISK_DIMENSIONS:
        tool_func = _DIM_TOOL_MAP[dim_cfg["tool"]]         # 根据配置找到对应工具
        tool_input = dim_cfg["tool_kwargs"](                # lambda 自动选择参数
            sentiment_result, financial
        )
        dim_result = tool_func.invoke(tool_input)           # 执行评估
        dimension_scores.append(dim_result)

    # ── 2. 加权综合 = Σ(维度评分 × 权重) ──
    overall = sum(
        d.score * cfg["weight"]
        for d, cfg in zip(dimension_scores, RISK_DIMENSIONS)
    )
    overall = round(min(overall, 1.0), 2)

    # ── 3. 构建推导链条（每步可追溯）──
    chain_parts = [
        f"{cfg['name']}({d.score}): {d.reasoning}"
        for d, cfg in zip(dimension_scores, RISK_DIMENSIONS)
    ]
    weight_detail = " + ".join(
        f"{cfg['weight']:.2f}×{d.score}"
        for d, cfg in zip(dimension_scores, RISK_DIMENSIONS)
    )
    chain_parts.append(f"加权综合: {weight_detail} = {overall}")

    # ── 4. 关键风险汇总（过滤掉"未发现"的占位 evidence）──
    key_risks: list[str] = []
    for dim in dimension_scores:
        for ev in dim.evidence:
            if ev and "未发现" not in ev:
                key_risks.append(f"[{dim.dimension}] {ev}")

    # ── 5. 等级判定 ──
    level = _score_to_level(overall)

    return RiskAssessment(
        symbol=symbol,
        company_name=company_name,
        overall_risk_level=level,
        overall_score=overall,
        dimensions=dimension_scores,
        risk_summary=(
            f"{company_name}({symbol}) 综合风险评分 {overall}，"
            f"判定为 {level.value.upper()} 风险。"
            f"共识别 {len(key_risks)} 项风险信号。"
        ),
        key_risks=key_risks,
        reasoning_chain="\n  → ".join(chain_parts),
    )


def get_risk_tools() -> list:
    """
    返回 Risk Agent 可用的风险评估工具列表。

    四个工具按粒度排列：
      1-3. 单维度评估（可独立调用）
      4.   综合评估（内部调 1-3 + 加权 + 等级判定）
    """
    return [assess_sentiment_risk, assess_financial_risk, assess_industry_risk, synthesize_risk]
