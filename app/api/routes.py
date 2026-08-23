"""
Sentiment & Risk Agent — FastAPI 路由

端点:
  GET  /api/v1/health              → 健康检查
  POST /api/v1/sentiment           → 舆情分析
  POST /api/v1/risk                → 风险评估（一站式：舆情→风险）
  POST /api/v1/sentiment-risk/full → 完整链路：舆情 → 风险 → 联合输出
"""

from fastapi import APIRouter

from app.agents.risk_agent import RiskAgent
from app.agents.sentiment_agent import SentimentAgent
from app.api.schemas import (
    APIResponse,
    RiskRequest,
    SentimentRequest,
)
from app.models.sentiment_risk_models import (
    FinancialSummary,
    SentimentInput,
    SentimentRiskJointOutput,
)

router = APIRouter(prefix="/api/v1", tags=["Sentiment & Risk"])


# ════════════════════════════════════════════════════════════
# Mock LLM（默认不依赖外部 API Key，可随时切换为真实 LLM）
# ════════════════════════════════════════════════════════════

class _MockLLM:
    """不调真实 API，返回固定文本。测试阶段默认使用。"""

    async def ainvoke(self, msg: str):
        class _Resp:
            content = "该企业近期舆情总体平稳，未发现显著负面信号。"
        return _Resp()


def _get_llm():
    """
    获取 LLM 实例。

    优先级：
      1. 如果 .env 里配了 DEEPSEEK_API_KEY → 用真实 DeepSeek
      2. 否则 → 用 Mock LLM（保证不配 Key 也能跑通）
    """
    try:
        from app.core.config import get_settings
        from app.core.llm_factory import get_llm
        settings = get_settings()
        if settings.deepseek_api_key:
            return get_llm("sentiment")
    except Exception:  # noqa: BLE001, S110 —— 尝试真实 LLM 失败时静默降级 Mock，保证不配 Key 可跑通
        pass
    return _MockLLM()


# ════════════════════════════════════════════════════════════
# 端点
# ════════════════════════════════════════════════════════════

@router.get("/health")
async def health_check():
    """健康检查——前端 / Nginx / Docker 用"""
    return {"status": "ok", "module": "sentiment-risk"}


@router.post("/sentiment", response_model=APIResponse)
async def analyze_sentiment(req: SentimentRequest):
    """
    舆情分析接口。

    输入：股票代码 + 企业名 + 回溯天数
    输出：新闻情感分布、热点主题、分析摘要

    示例请求：
        {"symbol": "300750", "company_name": "宁德时代", "days": 30}
    """
    try:
        llm = _get_llm()
        agent = SentimentAgent(llm=llm)
        result = await agent.run(
            SentimentInput(
                symbol=req.symbol,
                company_name=req.company_name,
                days=req.days,
            )
        )
        return APIResponse(
            success=True,
            message=f"舆情分析完成，共抓取 {result.searched_news_count} 条新闻",
            data=result.model_dump(),
        )
    except Exception as e:  # noqa: BLE001 —— 端点统一异常捕获，返回友好错误信封
        return APIResponse(success=False, message=str(e), data={})


@router.post("/risk", response_model=APIResponse)
async def assess_risk(req: RiskRequest):
    """
    风险评估接口（一站式：自动先跑舆情再跑风险）。

    输入：股票代码 + 企业名 + 可选财务数据
    输出：风险等级、三维度评分、关键风险项、推导链条

    示例请求：
        {
          "symbol": "300750",
          "company_name": "宁德时代",
          "days": 30,
          "revenue_growth": 0.15,
          "gross_margin": 0.22,
          "debt_ratio": 0.65,
          "anomalies": ["应收账款周转天数同比增加30%"]
        }
    """
    try:
        llm = _get_llm()

        # Step 1: 舆情分析
        s_agent = SentimentAgent(llm=llm)
        sentiment_result = await s_agent.run(
            SentimentInput(
                symbol=req.symbol,
                company_name=req.company_name,
                days=req.days,
            )
        )

        # Step 2: 风险评估
        financial = FinancialSummary(
            revenue_growth=req.revenue_growth,
            gross_margin=req.gross_margin,
            net_profit_margin=req.net_profit_margin,
            debt_ratio=req.debt_ratio,
            free_cash_flow=req.free_cash_flow,
            anomalies=req.anomalies or [],
        )
        r_agent = RiskAgent(llm=llm)
        risk_result = await r_agent.run(
            sentiment_result=sentiment_result,
            financial=financial,
        )

        return APIResponse(
            success=True,
            message=f"风险评估完成，等级: {risk_result.overall_risk_level.value.upper()}",
            data=risk_result.model_dump(),
        )
    except Exception as e:  # noqa: BLE001 —— 端点统一异常捕获，返回友好错误信封
        return APIResponse(success=False, message=str(e), data={})


@router.post("/sentiment-risk/full", response_model=APIResponse)
async def full_pipeline(req: RiskRequest):
    """
    完整链路：舆情分析 → 风险评估 → 联合输出。

    和 /risk 的区别：返回值同时包含 sentiment_result 和 risk_result，
    前端可以渲染完整的舆情详情 + 风险评估两张卡片。

    示例请求：同 /risk
    """
    try:
        llm = _get_llm()

        # Step 1: 舆情分析
        s_agent = SentimentAgent(llm=llm)
        sentiment_result = await s_agent.run(
            SentimentInput(
                symbol=req.symbol,
                company_name=req.company_name,
                days=req.days,
            )
        )

        # Step 2: 风险评估
        financial = FinancialSummary(
            revenue_growth=req.revenue_growth,
            gross_margin=req.gross_margin,
            net_profit_margin=req.net_profit_margin,
            debt_ratio=req.debt_ratio,
            free_cash_flow=req.free_cash_flow,
            anomalies=req.anomalies or [],
        )
        r_agent = RiskAgent(llm=llm)
        risk_result = await r_agent.run(
            sentiment_result=sentiment_result, financial=financial,
        )

        # Step 3: 联合输出
        joint = SentimentRiskJointOutput(
            sentiment=sentiment_result,
            risk=risk_result,
        )

        return APIResponse(
            success=True,
            message="完整分析完成",
            data=joint.model_dump(),
        )
    except Exception as e:  # noqa: BLE001 —— 端点统一异常捕获，返回友好错误信封
        return APIResponse(success=False, message=str(e), data={})
