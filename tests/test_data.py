"""
测试数据集：覆盖 Sentiment & Risk Agent 的各种场景。

每条数据都是一个完整的测试输入 + 预期输出，可以直接喂给 Agent。
按风险等级分组：LOW / MEDIUM / HIGH，外加边界情况。
"""

from datetime import datetime, timedelta

from app.models.sentiment_risk_models import (
    FinancialSummary,
    NewsItem,
    RiskLevel,
    ScoredNews,
    SentimentLabel,
    SentimentResult,
    SentimentScore,
    TopicCluster,
)

# ════════════════════════════════════════════════════════════
# 新闻样本
# ════════════════════════════════════════════════════════════

def _now(): return datetime.now()  # noqa: DTZ005 —— 测试样本相对时间，naive 足够


# ── 正面新闻 ──
NEWS_POSITIVE = [
    NewsItem(title="宁德时代发布第三代钠离子电池，能量密度突破 200Wh/kg",
             source="东方财富", url="https://example.com/n1",
             published_at=_now() - timedelta(days=1),
             summary="技术突破，产品竞争力提升，市场反应积极。"),
    NewsItem(title="宁德时代获宝马百亿欧元订单，欧洲市场份额持续扩大",
             source="财联社", url="https://example.com/n2",
             published_at=_now() - timedelta(days=2),
             summary="海外大单落地，营收增长确定性强。"),
    NewsItem(title="机构调研：宁德时代全球市占率升至38%，稳居第一",
             source="巨潮资讯", url="https://example.com/n3",
             published_at=_now() - timedelta(days=3),
             summary="行业地位稳固，护城河持续加深。"),
]

# ── 中性新闻 ──
NEWS_NEUTRAL = [
    NewsItem(title="宁德时代召开年度股东大会，审议利润分配方案",
             source="巨潮资讯", url="https://example.com/n4",
             published_at=_now() - timedelta(days=2),
             summary="例行会议，无重大事项披露。"),
    NewsItem(title="宁德时代披露ESG报告，碳排放强度同比下降12%",
             source="东方财富", url="https://example.com/n5",
             published_at=_now() - timedelta(days=4),
             summary="ESG表现改善，但不直接影响短期业绩。"),
    NewsItem(title="宁德时代与某高校签署产学研合作协议",
             source="财联社", url="https://example.com/n6",
             published_at=_now() - timedelta(days=5),
             summary="常规合作，布局前沿技术储备。"),
]

# ── 负面新闻 ──
NEWS_NEGATIVE = [
    NewsItem(title="欧盟对中国动力电池启动反补贴调查，宁德时代为主要目标",
             source="财联社", url="https://example.com/n7",
             published_at=_now() - timedelta(days=1),
             summary="欧盟贸易保护升级，或面临额外关税，海外营收承压。"),
    NewsItem(title="碳酸锂价格暴涨30%，宁德时代毛利率或进一步承压",
             source="东方财富", url="https://example.com/n8",
             published_at=_now() - timedelta(days=2),
             summary="上游原材料涨价，成本端压力加大。"),
    NewsItem(title="宁德时代被列入美国国防部涉军企业清单",
             source="财联社", url="https://example.com/n9",
             published_at=_now() - timedelta(days=3),
             summary="地缘政治风险加剧，海外业务不确定性上升。"),
    NewsItem(title="固态电池企业QuantumScape宣布重大突破，传统锂电路线受质疑",
             source="东方财富", url="https://example.com/n10",
             published_at=_now() - timedelta(days=4),
             summary="下一代电池技术竞争加剧，技术替代风险。"),
    NewsItem(title="宁德时代核心技术人员张某离职，加盟竞争对手",
             source="财联社", url="https://example.com/n11",
             published_at=_now() - timedelta(days=5),
             summary="核心人才流失，研发团队稳定性存疑。"),
    NewsItem(title="因价格战加剧，宁德时代Q3电池单价同比下降15%",
             source="东方财富", url="https://example.com/n12",
             published_at=_now() - timedelta(days=6),
             summary="行业竞争白热化，以价换量策略可持续性存疑。"),
]

# ── 高危主题新闻（监管+制裁+关税） ──
NEWS_HIGH_RISK = [
    NewsItem(title="美国商务部拟对华动力电池加征100%惩罚性关税",
             source="财联社", url="https://example.com/h1",
             published_at=_now() - timedelta(days=1),
             summary="极端关税政策将严重冲击中国电池企业对美出口。"),
    NewsItem(title="欧洲议会通过《关键原材料法案》，限制中国电池供应链",
             source="东方财富", url="https://example.com/h2",
             published_at=_now() - timedelta(days=1),
             summary="欧盟立法限制对中国电池材料的依赖，供应链面临重构。"),
    NewsItem(title="宁德时代德国工厂因环保诉讼被勒令停产整改",
             source="财联社", url="https://example.com/h3",
             published_at=_now() - timedelta(days=2),
             summary="海外生产基地遭遇法律挑战，产能释放推迟。"),
    NewsItem(title="中国证监会因信息披露问题对宁德时代立案调查",
             source="巨潮资讯", url="https://example.com/h4",
             published_at=_now() - timedelta(days=2),
             summary="监管风险骤增，可能面临罚款和投资者索赔。"),
]


# ════════════════════════════════════════════════════════════
# 预构建的 SentimentResult（跳过 Phase 1 工具链，直接喂给 Risk Agent）
# ════════════════════════════════════════════════════════════

def make_sentiment_result(symbol: str, company: str,
                          news: list[NewsItem],
                          labels: list[SentimentLabel]) -> SentimentResult:
    """快速构建 SentimentResult——用于测试 Risk Agent 时跳过 Phase 1"""
    scored = [
        ScoredNews(
            news=n,
            sentiment=SentimentScore(
                label=label, confidence=0.85,
                explanation=f"FinBERT判为{label.value}"
            )
        )
        for n, label in zip(news, labels)
    ]
    dist = {"positive": 0, "negative": 0, "neutral": 0}
    for s in scored:
        dist[s.sentiment.label.value] += 1
    return SentimentResult(
        symbol=symbol, company_name=company,
        searched_news_count=len(scored),
        scored_news=scored,
        sentiment_distribution=dist,
        topics=[], summary="",
    )


# ── 场景 1：低风险 — 舆情平稳、财务健康 ──
SENTIMENT_LOW_RISK = make_sentiment_result(
    "300750", "宁德时代",
    NEWS_POSITIVE + NEWS_NEUTRAL,
    [SentimentLabel.POSITIVE] * 3 + [SentimentLabel.NEUTRAL] * 3,
)
SENTIMENT_LOW_RISK.topics = [
    TopicCluster(topic_id=0, label="技术创新", keywords=["钠离子电池","产能"],
                 news_count=2, representative_news=["宁德时代发布第三代钠离子电池..."]),
    TopicCluster(topic_id=1, label="ESG与治理", keywords=["ESG","股东大会"],
                 news_count=1, representative_news=["宁德时代披露ESG报告..."]),
]
SENTIMENT_LOW_RISK.summary = "近期舆情总体积极，技术创新主题突出，机构关注度高。"

FINANCIAL_HEALTHY = FinancialSummary(
    revenue_growth=0.25,
    gross_margin=0.28,
    net_profit_margin=0.12,
    debt_ratio=0.55,
    free_cash_flow=85.0,
    anomalies=[],
)

# ── 场景 2：中风险 — 部分负面舆情 + 财务指标承受压力 ──
SENTIMENT_MEDIUM_RISK = make_sentiment_result(
    "300750", "宁德时代",
    NEWS_POSITIVE[:1] + NEWS_NEUTRAL[:1] + NEWS_NEGATIVE[:4],
    [SentimentLabel.NEUTRAL, SentimentLabel.NEUTRAL,
     SentimentLabel.NEGATIVE, SentimentLabel.NEGATIVE,
     SentimentLabel.NEGATIVE, SentimentLabel.NEGATIVE],
)
SENTIMENT_MEDIUM_RISK.topics = [
    TopicCluster(topic_id=0, label="海外关税风险", keywords=["欧盟","反补贴","关税"],
                 news_count=3, representative_news=["欧盟对中国动力电池启动反补贴调查..."]),
    TopicCluster(topic_id=1, label="成本端压力", keywords=["碳酸锂","毛利"],
                 news_count=2, representative_news=["碳酸锂价格暴涨30%..."]),
    TopicCluster(topic_id=2, label="竞争格局恶化", keywords=["价格战","固态电池"],
                 news_count=2, representative_news=["因价格战加剧..."]),
]
SENTIMENT_MEDIUM_RISK.summary = "负面舆情占比超过50%，海外关税和成本端压力为关注焦点。"

FINANCIAL_UNDER_PRESSURE = FinancialSummary(
    revenue_growth=0.03,
    gross_margin=0.18,
    net_profit_margin=0.05,
    debt_ratio=0.62,
    free_cash_flow=30.0,
    anomalies=[
        "应收账款周转天数从45天增至68天，回款效率下降",
        "存货周转天数同比增加22天，库存积压风险",
    ],
)

# ── 场景 3：高风险 — 监管制裁叠加 + 财务恶化 ──
SENTIMENT_HIGH_RISK = make_sentiment_result(
    "300750", "宁德时代",
    NEWS_HIGH_RISK + NEWS_NEGATIVE[:2],
    [SentimentLabel.NEGATIVE] * 6,
)
SENTIMENT_HIGH_RISK.topics = [
    TopicCluster(topic_id=0, label="美国关税制裁", keywords=["关税","制裁","100%"],
                 news_count=2, representative_news=["美国商务部拟对华动力电池加征100%惩罚性关税..."]),
    TopicCluster(topic_id=1, label="欧洲监管与供应链限制", keywords=["欧盟","供应链","监管"],
                 news_count=2, representative_news=["欧洲议会通过《关键原材料法案》..."]),
    TopicCluster(topic_id=2, label="诉讼与合规风险", keywords=["诉讼","调查","停产"],
                 news_count=2, representative_news=["宁德时代德国工厂因环保诉讼被勒令停产..."]),
]
SENTIMENT_HIGH_RISK.summary = "高危：美欧监管制裁叠加，海外业务面临颠覆性风险；同时面临证监会调查。"

FINANCIAL_CRITICAL = FinancialSummary(
    revenue_growth=-0.08,
    gross_margin=0.12,
    net_profit_margin=0.01,
    debt_ratio=0.78,
    free_cash_flow=-15.0,
    anomalies=[
        "营收同比下滑8%，为近五年首次负增长",
        "自由现金流转负，经营活动现金流无法覆盖资本支出",
        "资产负债率升至78%，短期借款同比增加45%",
        "审计师出具带强调事项段的无保留意见",
    ],
)

# ── 场景 4：边界情况 — 空数据 ──
SENTIMENT_EMPTY = SentimentResult(
    symbol="000001", company_name="未知企业",
    searched_news_count=0, scored_news=[],
    sentiment_distribution={"positive": 0, "negative": 0, "neutral": 0},
    topics=[], summary="未找到任何新闻。",
)
FINANCIAL_EMPTY = FinancialSummary()

# ── 场景 5：边界情况 — 完全正面 ──
SENTIMENT_ALL_POSITIVE = make_sentiment_result(
    "600519", "贵州茅台",
    NEWS_POSITIVE * 2,     # 6 条正面新闻
    [SentimentLabel.POSITIVE] * 6,
)
SENTIMENT_ALL_POSITIVE.topics = [
    TopicCluster(topic_id=0, label="业绩增长", keywords=["营收","利润"],
                 news_count=3, representative_news=["..."])
]
SENTIMENT_ALL_POSITIVE.summary = "舆情全面向好，无负面信号。"
FINANCIAL_STRONG = FinancialSummary(
    revenue_growth=0.18, gross_margin=0.92, net_profit_margin=0.52,
    debt_ratio=0.19, free_cash_flow=200.0, anomalies=[],
)


# ════════════════════════════════════════════════════════════
# 测试用例元数据（场景名 → 输入 + 预期）
# ════════════════════════════════════════════════════════════

TEST_SCENARIOS = {
    "low_risk": {
        "name": "低风险场景——宁德时代舆情平稳、财务健康",
        "sentiment": SENTIMENT_LOW_RISK,
        "financial": FINANCIAL_HEALTHY,
        "expect_level": RiskLevel.LOW,
        "expect_max_score": 0.39,
        "expect_min_key_risks": 0,
    },
    "medium_risk": {
        "name": "中风险场景——宁德时代负面舆情过半、财务承压",
        "sentiment": SENTIMENT_MEDIUM_RISK,
        "financial": FINANCIAL_UNDER_PRESSURE,
        "expect_level": RiskLevel.MEDIUM,
        "expect_min_score": 0.35,
        "expect_min_key_risks": 3,
    },
    "high_risk": {
        "name": "高风险场景——宁德时代监管制裁+财务恶化",
        "sentiment": SENTIMENT_HIGH_RISK,
        "financial": FINANCIAL_CRITICAL,
        "expect_level": RiskLevel.HIGH,
        "expect_min_score": 0.65,
        "expect_min_key_risks": 5,
    },
    "empty_data": {
        "name": "边界——空数据",
        "sentiment": SENTIMENT_EMPTY,
        "financial": FINANCIAL_EMPTY,
        "expect_level": RiskLevel.LOW,
        "expect_max_score": 0.10,
        "expect_min_key_risks": 0,
    },
    "all_positive": {
        "name": "边界——完全正面舆情+极强财务",
        "sentiment": SENTIMENT_ALL_POSITIVE,
        "financial": FINANCIAL_STRONG,
        "expect_level": RiskLevel.LOW,
        "expect_max_score": 0.15,
        "expect_min_key_risks": 0,
    },
}
