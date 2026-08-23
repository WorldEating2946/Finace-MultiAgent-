"""
Risk Agent — 综合风险推理智能体

════════════════════ 职责 ════════════════════════
融合舆情信号（SentimentResult）+ 财务数据（FinancialSummary）+ 行业周期
→ 三维度独立评分 → 加权综合 → 风险等级判定 → LLM 润色总结

════════════════════ 输入/输出 ════════════════════════
输入: SentimentResult + FinancialSummary（两个上游 Agent 的输出）
输出: RiskAssessment（含多维度评分、关键风险项、完整推导链条）

════════════════════ 设计原则 ════════════════════════
1. 独立可测试——不依赖 LangGraph，直接 agent.run(sentiment_result, financial) 就能跑。

2. 评分逻辑在 @tool 函数里（risk_tools.py）——纯 Python 计算，确定性强，可复现。
   Agent 只负责编排：调工具 → LLM 润色 → 返回。

3. LLM 只润色风险总结的叙事质量，不参与评分决策。
   评分逻辑出错 = 改代码；叙事不好 = LLM 改。职责分离。

4. 上游数据缺失的容错——FinancialSummary 所有字段都是 Optional。
   工具函数逐项 is not None 检查，不会因为缺字段崩溃。
═══════════════════════════════════════════════════════════
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ..models.sentiment_risk_models import (
    FinancialSummary,
    RiskAssessment,
    SentimentResult,
)
from ..tools.risk_tools import synthesize_risk

# ── System Prompt ─────────────────────────────────────────
# Risk Agent 的角色定义和评估框架。Phase 1 主要用于 LLM 润色时提供角色上下文。
# Phase 2 切换 ReAct Agent 后，作为 SystemMessage 指导 LLM 自主编排评估流程。
RISK_AGENT_PROMPT = """你是一位资深金融风控分析师。你的任务是基于多源信号对目标企业做综合风险评估。

## 你的工具
- synthesize_risk: 融合舆情、财务、行业三维度信号进行综合评分和等级判定

## 分析框架
你将收到两份上游数据:
1. 舆情分析结果(SentimentResult): 新闻情感分布、热点主题
2. 财务数据摘要(FinancialSummary): 核心财务指标、异常项

你需要从以下维度评估:
- **舆情负面信号**(权重40%): 负面新闻占比、高危主题出现频率
- **财务异常信号**(权重35%): 营收/利润趋势、负债水平、数据异常
- **行业周期风险**(权重25%): 政策监管、供应链、竞争格局、技术替代

## 输出要求
- 风险等级判定必须有理有据，推理链条可追溯
- 关键风险项按严重程度排序
- 避免空洞的泛化评价，每个结论需对应具体证据
"""


class RiskAgent:
    """
    风险评估 Agent。

    ══════════════════ 使用方式 ══════════════════
    from app.core.llm_factory import get_llm
    from app.models.sentiment_risk_models import FinancialSummary

    llm = get_llm("risk")
    agent = RiskAgent(llm=llm)
    result = await agent.run(
        sentiment_result=sentiment_agent_result,   # from SentimentAgent.run()
        financial=FinancialSummary(
            revenue_growth=0.15,
            gross_margin=0.22,
            debt_ratio=0.65,
            anomalies=["应收账款周转天数同比增加30%"],
        )
    )
    # result.overall_risk_level → RiskLevel.MEDIUM
    # result.reasoning_chain → 完整推导链条
    ════════════════════════════════════════════════
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.tools: list[BaseTool] = [synthesize_risk]

    @property
    def prompt(self) -> str:
        return RISK_AGENT_PROMPT

    async def run(
        self,
        sentiment_result: SentimentResult,
        financial: FinancialSummary,
    ) -> RiskAssessment:
        """
        执行综合风险评估主流程。

        两步走：
          Step 1: 调用 synthesize_risk 工具做结构化评分
                  → 纯 Python 计算（在 risk_tools.py 里），LLM 不参与评分
          Step 2: 用 LLM 润色风险总结
                  → LLM 基于评分结果写 200 字自然语言总结

        为什么 LLM 不参与评分？
          评分需要确定性和可复现性——同样的数据，每次评分必须一致。
          LLM 有随机性（即使 temperature=0），可能同一份数据两次评分不同。
          评分交给代码（纯规则 + 加权），叙事交给 LLM。各司其职。
        """
        # ── Step 1: 结构化风险评分（纯 Python，无 LLM） ──
        raw_assessment = synthesize_risk.invoke({
            "symbol": sentiment_result.symbol,
            "company_name": sentiment_result.company_name,
            "sentiment_result": sentiment_result,
            "financial": financial,
        })

        # ── Step 2: LLM 润色总结 ──
        enhanced = await self._enhance_with_llm(raw_assessment)

        return enhanced

    async def _enhance_with_llm(self, assessment: RiskAssessment) -> RiskAssessment:
        """
        使用 LLM 增强风险报告的叙事质量。

        输入给 LLM 的不是原始数据，而是已完成的结构化评分——
        LLM 的工作是把评分数据写成人类易读的 200 字总结。

        格式：维度评分 + 证据列表 + 综合评分 → LLM → 自然语言总结
        """
        # ── 组装评分上下文 ──
        dim_lines = "\n".join(
            f"[{d.dimension}] 评分 {d.score}: {d.reasoning}\n  证据: {'; '.join(d.evidence)}"
            for d in assessment.dimensions
        )
        risks_str = (
            "\n".join(f"- {r}" for r in assessment.key_risks)
            if assessment.key_risks
            else "- 未识别明确风险项"
        )

        user_msg = (
            f"你是金融风控分析师。请基于以下评分结果，"
            f"为 {assessment.company_name}({assessment.symbol}) "
            f"写一段 200 字以内的风险总结：\n\n"
            f"综合评分: {assessment.overall_score} → 等级 {assessment.overall_risk_level.value.upper()}\n"
            f"维度评分:\n{dim_lines}\n\n"
            f"已识别风险:\n{risks_str}\n\n"
            f"要求: 1-2 句话概括主要风险来源，避免重复已有评分数据。"
        )
        resp = await self.llm.ainvoke(user_msg)
        summary_text = str(resp.content) if hasattr(resp, "content") else str(resp)

        if summary_text:
            assessment.risk_summary = summary_text

        return assessment
