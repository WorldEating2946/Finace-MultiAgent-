"""
ReportAssembler 单元测试

覆盖:
  1. 六章结构完整（key/title 顺序正确）
  2. 财务指标落入第二章表格
  3. 缺 sentiment → 第四章保留且「（数据待补充）」
  4. 全 None 输入不崩
  5. research 占位形态（business_model/industry_position）兼容
  6. 同一输入两次 assemble → 内容确定性一致
  7. 第六章恒含免责声明
"""

import pytest

from app.report.assembler import ReportAssembler
from app.report.schemas import ReportContent

FINANCIAL = {
    "company": "宁德时代",
    "ticker": "300750",
    "analysis_period": "2021年报",
    "key_metrics": {
        "roe_pct": 21.5,
        "net_profit_margin_pct": 13.7,
        "revenue_yoy_pct": 159.1,
        "net_profit_yoy_pct": 185.3,
        "equity_multiplier": 2.1,
        "asset_turnover": 0.68,
    },
    "dupont": {
        "net_profit_margin": 13.7,
        "asset_turnover": 0.68,
        "equity_multiplier": 2.1,
        "roe_computed": 19.6,
        "roe_direct": 21.5,
    },
    "yoy_history": [
        {"period": "2020", "revenue_growth_pct": 9.9, "revenue_trend": "增长", "net_profit_growth_pct": 22.4, "profit_trend": "增长"},
        {"period": "2021", "revenue_growth_pct": 159.1, "revenue_trend": "大幅增长", "net_profit_growth_pct": 185.3, "profit_trend": "大幅增长"},
    ],
    "commentary": "营收利润双高增，盈利质量较好。",
    "data_source": "akshare",
    "fetch_error": None,
}

SENTIMENT = {
    "symbol": "300750",
    "company_name": "宁德时代",
    "searched_news_count": 12,
    "sentiment_distribution": {"positive": 6, "negative": 4, "neutral": 2},
    "topics": [
        {"topic_id": 0, "label": "产品与技术", "keywords": ["麒麟电池"], "news_count": 5, "representative_news": ["..."]},
    ],
    "summary": "整体情绪中性偏多。",
}

RISK = {
    "symbol": "300750",
    "company_name": "宁德时代",
    "overall_risk_level": "HIGH",
    "overall_score": 0.75,
    "dimensions": [
        {"dimension": "市场风险", "score": 0.8, "evidence": ["估值偏高"], "reasoning": "估值处于高位"},
        {"dimension": "经营风险", "score": 0.5, "evidence": [], "reasoning": "行业竞争加剧"},
    ],
    "key_risks": ["原材料价格波动", "行业竞争加剧"],
    "reasoning_chain": "市场风险 > 经营风险 → 综合高风险",
    "risk_summary": "综合风险较高。",
}

RESEARCH_REPORT = {
    "title": "宁德时代研究报告",
    "summary": "全球动力电池龙头，技术与规模领先。",
    "plan_summary": "",
    "advantages": [{"claim": "全球市占率第一"}],
    "risks": [],
    "uncertainties": [],
    "evidence": [],
}

RESEARCH_PLACEHOLDER = {
    "company": "宁德时代",
    "summary": "动力电池行业龙头。",
    "business_model": "研发-生产-销售一体化",
    "industry_position": "全球动力电池装机量第一",
    "competitive_advantages": ["技术壁垒高", "客户绑定深"],
}


@pytest.fixture
def assembler() -> ReportAssembler:
    return ReportAssembler()


def _text_of(content: ReportContent, key: str) -> str:
    sec = next(s for s in content.sections if s.key == key)
    parts: list[str] = []
    for b in sec.blocks:
        if b.text:
            parts.append(b.text)
        parts.extend(b.items)
        for row in b.rows:
            parts.extend(row)
    return "\n".join(parts)


def test_full_content_six_chapters(assembler):
    content = assembler.assemble(
        company="宁德时代",
        ticker="300750",
        research=RESEARCH_REPORT,
        financial=FINANCIAL,
        sentiment=SENTIMENT,
        risk=RISK,
    )
    keys = [s.key for s in content.sections]
    assert keys == [
        "chapter_1",
        "chapter_2",
        "chapter_3",
        "chapter_4",
        "chapter_5",
        "chapter_6",
    ]
    assert content.title == "宁德时代 深度投研分析报告"
    assert content.company == "宁德时代"
    assert "一、企业概况（300750）" in [s.title for s in content.sections]


def test_financial_metrics_in_table(assembler):
    content = assembler.assemble(company="宁德时代", financial=FINANCIAL)
    fin = next(s for s in content.sections if s.key == "chapter_2")
    tables = [b for b in fin.blocks if b.kind == "table"]
    assert tables, "财务章节应含表格"
    assert "21.5%" in _text_of(content, "chapter_2")
    assert "159.1%" in _text_of(content, "chapter_2")
    assert "2021年报" in _text_of(content, "chapter_2")
    assert "akshare" in _text_of(content, "chapter_2")


def test_missing_sentiment_falls_back(assembler):
    content = assembler.assemble(company="宁德时代", financial=FINANCIAL)
    sec = next(s for s in content.sections if s.key == "chapter_4")
    assert sec.blocks and sec.blocks[0].text == "（数据待补充）"


def test_all_none_no_crash(assembler):
    content = assembler.assemble(company="测试公司")
    assert len(content.sections) == 6
    # 前五章数据缺失 → 待补充；第六章为规则建议，恒有内容
    for sec in content.sections[:5]:
        assert sec.blocks and sec.blocks[0].text == "（数据待补充）"


def test_research_placeholder_compat(assembler):
    content = assembler.assemble(company="宁德时代", research=RESEARCH_PLACEHOLDER)
    txt = _text_of(content, "chapter_1")
    assert "动力电池行业龙头" in txt
    assert "研发-生产-销售一体化" in txt
    assert "全球动力电池装机量第一" in txt
    assert "技术壁垒高" in txt
    # 行业与竞争力章复用同一字段
    ind = _text_of(content, "chapter_3")
    assert "全球动力电池装机量第一" in ind


def test_risk_level_chinese(assembler):
    content = assembler.assemble(company="宁德时代", risk=RISK)
    txt = _text_of(content, "chapter_5")
    assert "高风险" in txt
    assert "0.750" in txt
    assert "原材料价格波动" in txt
    # 投资建议章节引用高风险
    adv = _text_of(content, "chapter_6")
    assert "谨慎" in adv


def test_advice_always_has_disclaimer(assembler):
    content = assembler.assemble(company="宁德时代")
    adv = _text_of(content, "chapter_6")
    assert "免责声明" in adv
    assert "不构成任何投资建议" in adv


def test_deterministic_except_timestamp(assembler):
    a = assembler.assemble(
        company="宁德时代", financial=FINANCIAL, sentiment=SENTIMENT, risk=RISK
    )
    b = assembler.assemble(
        company="宁德时代", financial=FINANCIAL, sentiment=SENTIMENT, risk=RISK
    )
    # 同一输入两次组装 → 内容完全一致；时间戳仅需非空 ISO 格式
    assert a.model_dump(exclude={"generated_at"}) == b.model_dump(exclude={"generated_at"})
    assert a.generated_at.startswith("202") and "T" in a.generated_at
    assert b.generated_at.startswith("202") and "T" in b.generated_at
