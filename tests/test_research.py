"""研究意图理解与规划测试（PR #36）。

覆盖：schema / IntentParser（意图分类、公司/板块抽取、维度推导）/ ResearchPlanner
（模板步骤、{segment} 省略）/ build_research_plan 端到端。
纯规划层，无数据库/网络依赖。
"""

from app.rag.research import (
    IntentParser,
    ResearchDimension,
    ResearchIntent,
    ResearchPlanner,
    ResearchTarget,
    build_research_plan,
)
from app.rag.research.schema import ResearchPlan


# ── Schema ─────────────────────────────────────────────────────

def test_research_plan_schema():
    """ResearchPlan 结构化正确。"""
    plan = ResearchPlan(
        request="分析小米汽车未来竞争力",
        intent=ResearchIntent.COMPETITIVE_ANALYSIS,
        target=ResearchTarget(company="小米", segment="汽车"),
        dimensions=[ResearchDimension.COMPETITION],
    )
    assert plan.intent.value == "competitive_analysis"
    assert plan.target.company == "小米"
    assert plan.target.segment == "汽车"


def test_intent_enum_values():
    """ResearchIntent 枚举值与意图一致。"""
    assert ResearchIntent.COMPETITIVE_ANALYSIS.value == "competitive_analysis"
    assert ResearchIntent.FINANCIAL_ANALYSIS.value == "financial_analysis"
    assert ResearchIntent.GENERIC_RESEARCH.value == "generic_research"


# ── IntentParser：意图分类 ─────────────────────────────────────

def test_classify_competitive():
    """竞争力关键词 → COMPETITIVE_ANALYSIS。"""
    p = IntentParser()
    intent, target, _ = p.parse("分析小米汽车未来竞争力")
    assert intent == ResearchIntent.COMPETITIVE_ANALYSIS


def test_classify_risk():
    assert IntentParser().parse("小米经营风险分析")[0] == ResearchIntent.RISK_ANALYSIS


def test_classify_financial():
    assert IntentParser().parse("宁德时代盈利能力和财务状况")[0] == ResearchIntent.FINANCIAL_ANALYSIS


def test_classify_policy():
    assert IntentParser().parse("新能源汽车补贴政策影响")[0] == ResearchIntent.POLICY_ANALYSIS


def test_classify_fallback_generic():
    """无关键词命中 → GENERIC_RESEARCH 兜底。"""
    assert IntentParser().parse("这个公司怎么样")[0] == ResearchIntent.GENERIC_RESEARCH


def test_classify_multikeyword_priority():
    """多意图关键词并存 → 命中数最多者胜（竞争>风险）。"""
    intent, _, _ = IntentParser().parse("小米竞争力与经营风险")
    assert intent == ResearchIntent.COMPETITIVE_ANALYSIS


# ── IntentParser：目标抽取 ─────────────────────────────────────

def test_extract_company_alias():
    """公司别名（宁德/CATL/xiaomi）→ 标准名。"""
    assert IntentParser().parse("宁德时代怎么转型")[1].company == "宁德时代"
    assert IntentParser().parse("catl 的电池业务")[1].company == "宁德时代"
    assert IntentParser().parse("xiaomi 国际化")[1].company == "小米"


def test_extract_company_missing():
    """无公司名 → 空。"""
    assert IntentParser().parse("这个行业前景如何")[1].company == ""


def test_extract_segment():
    """业务板块关键词 → segment。"""
    assert IntentParser().parse("小米汽车销量如何")[1].segment == "汽车"
    assert IntentParser().parse("智能手机出货量")[1].segment == "手机"
    assert IntentParser().parse("动力电池技术")[1].segment == "电池"
    assert IntentParser().parse("公司整体情况")[1].segment == ""


def test_infer_dimensions_automotive_adds_policy():
    """汽车 segment → 自动补 POLICY 维度。"""
    p = IntentParser()
    intent, target, dims = p.parse("小米汽车未来竞争力")
    assert intent == ResearchIntent.COMPETITIVE_ANALYSIS
    assert target.segment == "汽车"
    assert ResearchDimension.POLICY in dims
    # 去重保序
    assert len(dims) == len(set(dims))


# ── ResearchPlanner ────────────────────────────────────────────

def test_plan_generates_ordered_steps():
    """plan() 生成有序步骤，依赖前序。"""
    planner = ResearchPlanner()
    steps = planner.plan(
        ResearchIntent.COMPETITIVE_ANALYSIS,
        ResearchTarget(company="小米", segment="汽车"),
    )
    assert len(steps) >= 5
    assert [s.order for s in steps] == list(range(1, len(steps) + 1))
    assert all(s.depends_on == list(range(1, s.order)) for s in steps)
    # query 已替换
    assert "小米" in steps[0].retrieval_query
    assert "汽车" in steps[1].retrieval_query


def test_plan_omits_segment_when_empty():
    """segment 为空 → query 中不出现 segment 关键词。"""
    planner = ResearchPlanner()
    steps = planner.plan(
        ResearchIntent.COMPETITIVE_ANALYSIS,
        ResearchTarget(company="小米", segment=""),
    )
    # 步骤名与 query 都无 "汽车"
    for s in steps:
        assert "汽车" not in s.name
        assert "汽车" not in s.retrieval_query
    # 政策步骤 query 也退化为空 segment 主题
    policy_step = steps[5]  # 政策与监管环境
    assert "汽车" not in policy_step.retrieval_query


def test_plan_all_intents_have_templates():
    """每个 intent 都有步骤模板（非空）。"""
    planner = ResearchPlanner()
    for intent in ResearchIntent:
        steps = planner.plan(intent, ResearchTarget(company="测试"))
        assert steps, f"intent {intent} 无步骤模板"
        assert steps[0].order == 1


# ── build_research_plan 端到端 ─────────────────────────────────

def test_build_research_plan_end_to_end():
    """一步构建完整研究计划。"""
    plan = build_research_plan("分析小米汽车未来竞争力")

    assert isinstance(plan, ResearchPlan)
    assert plan.request == "分析小米汽车未来竞争力"
    assert plan.intent == ResearchIntent.COMPETITIVE_ANALYSIS
    assert plan.target.company == "小米"
    assert plan.target.segment == "汽车"
    assert len(plan.steps) >= 5
    assert 0.0 <= plan.confidence <= 1.0
    assert plan.created_at


def test_build_research_plan_fallback_confidence():
    """兜底意图 → 低置信度。"""
    plan = build_research_plan("随便看看")
    assert plan.intent == ResearchIntent.GENERIC_RESEARCH
    assert plan.confidence < 0.5


def test_build_research_plan_policy_source():
    """竞争力分析步骤含 policy source（政策维度）。"""
    plan = build_research_plan("小米汽车竞争力")
    assert any("policy" in s.source_types for s in plan.steps)
