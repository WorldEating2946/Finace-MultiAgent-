"""
Sentiment & Risk Agent 端到端链路测试

覆盖:
  1. 模块导入验证
  2. Sentiment Agent 完整流程（Mock 新闻数据）
  3. Risk Agent 5 个风险场景（低/中/高/空/全正面）
  4. 联合输出组装
"""

import pytest

from app.agents.risk_agent import RiskAgent
from app.agents.sentiment_agent import SentimentAgent
from app.models.sentiment_risk_models import (
    SentimentInput,
    SentimentRiskJointOutput,
)
from tests.test_data import TEST_SCENARIOS

# ── Mock LLM ──────────────────────────────────────────────

class _MockLLM:
    """占位 LLM，不调用 API，返回固定文本"""

    async def ainvoke(self, msg: str):
        class _Resp:
            content = "（测试摘要）该企业近期舆情总体平稳，未发现显著负面聚集。"
        return _Resp()


# ── 模块导入 ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_imports():
    """验证所有模块可正常导入"""
    from app.tools.news_tools import fetch_recent_news
    from app.tools.risk_tools import RISK_DIMENSIONS, synthesize_risk
    assert synthesize_risk is not None
    assert fetch_recent_news is not None
    assert len(RISK_DIMENSIONS) == 3, "风险维度配置表应为3个维度"


# ── Sentiment Agent 基础流程 ──────────────────────────────

@pytest.mark.asyncio
async def test_sentiment_agent():
    """验证 Sentiment Agent 完整流程：抓新闻→评分→聚类→摘要"""
    agent = SentimentAgent(llm=_MockLLM())
    result = await agent.run(
        SentimentInput(symbol="300750", company_name="宁德时代", days=30)
    )
    assert result.searched_news_count == 9
    assert result.sentiment_distribution["positive"] >= 0
    assert len(result.topics) >= 1
    assert len(result.summary) > 0


# ── Risk Agent 5 个场景 ──────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_key", TEST_SCENARIOS.keys())
async def test_risk_scenarios(scenario_key):
    """遍历 5 个风险场景，验证评分和等级"""
    s = TEST_SCENARIOS[scenario_key]
    agent = RiskAgent(llm=_MockLLM())
    result = await agent.run(
        sentiment_result=s["sentiment"],
        financial=s["financial"],
    )

    # 评分必须在 [0, 1]
    assert 0.0 <= result.overall_score <= 1.0, \
        f"场景{s['name']}：评分 {result.overall_score} 超出 [0,1]"

    # 等级必须合法
    assert result.overall_risk_level.value in ("high", "medium", "low"), \
        f"场景{s['name']}：非法等级 {result.overall_risk_level}"

    # 三维度必须齐全
    assert len(result.dimensions) == 3, \
        f"场景{s['name']}：维度数 {len(result.dimensions)} != 3"

    # 推导链不能空
    assert len(result.reasoning_chain) > 0, \
        f"场景{s['name']}：推导链为空"

    # ── 场景特定断言 ──
    if "expect_level" in s:
        assert result.overall_risk_level == s["expect_level"], \
            f"场景{s['name']}：期望 {s['expect_level'].value}，实际 {result.overall_risk_level.value}，评分 {result.overall_score}"

    if "expect_min_score" in s:
        assert result.overall_score >= s["expect_min_score"], \
            f"场景{s['name']}：期望评分 ≥ {s['expect_min_score']}，实际 {result.overall_score}"

    if "expect_max_score" in s:
        assert result.overall_score <= s["expect_max_score"], \
            f"场景{s['name']}：期望评分 ≤ {s['expect_max_score']}，实际 {result.overall_score}"

    if "expect_min_key_risks" in s:
        assert len(result.key_risks) >= s["expect_min_key_risks"], \
            f"场景{s['name']}：期望关键风险 ≥ {s['expect_min_key_risks']}，实际 {len(result.key_risks)}"


# ── 联合输出组装 ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_joint_output():
    """验证 SentimentResult + RiskAssessment 能正确打包为 JointOutput"""
    agent = SentimentAgent(llm=_MockLLM())
    sentiment_result = await agent.run(
        SentimentInput(symbol="300750", company_name="宁德时代", days=30)
    )
    risk_agent = RiskAgent(llm=_MockLLM())
    risk_result = await risk_agent.run(
        sentiment_result=sentiment_result,
        financial=TEST_SCENARIOS["medium_risk"]["financial"],
    )
    joint = SentimentRiskJointOutput(sentiment=sentiment_result, risk=risk_result)
    assert joint.sentiment is not None
    assert joint.risk is not None
    assert joint.sentiment.symbol == joint.risk.symbol
