"""
Sentiment Agent — 舆情分析智能体

════════════════════ 职责 ════════════════════════
实时新闻抓取 → FinBERT 情感评分 → BERTopic 热点聚类 → LLM 舆情摘要

════════════════════ 输入/输出 ════════════════════════
输入: SentimentInput（股票代码 + 企业名称 + 回溯天数）
输出: SentimentResult（情感分布 + 热点主题 + 分析摘要）

════════════════════ 设计原则 ════════════════════════
1. 独立可测试——不依赖 LangGraph，直接 agent.run(params) 就能跑。
   测试时传 _MockLLM，不需要真实 API Key。

2. 工具链串行调用——Step 1→2→3→4 有严格先后依赖：
   抓新闻 → 评分 → 聚类 → 摘要。爬完新闻才能评分，评完分才能聚类。

3. Phase 1 直接串行调用工具链，Phase 2 可切换为 LangGraph ReAct Agent，
   让 LLM 自主决定工具调用顺序。当前实现保证了两阶段的接口兼容。

4. 异常安全——新闻抓取返回 0 条时，不走评分/聚类，直接返回空结果 + 提示，
   不会因为空列表导致后续工具报错。
═══════════════════════════════════════════════════════════
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ..models.sentiment_risk_models import (
    SentimentInput,
    SentimentResult,
)
from ..tools.news_tools import fetch_recent_news
from ..tools.sentiment_tools import batch_score_news, cluster_topics

# ── System Prompt ─────────────────────────────────────────
# 这是 Agent 的「人设」和「操作手册」。
# Phase 1：仅用于 LLM 润色摘要时提供角色上下文。
# Phase 2：切换为 LangGraph ReAct Agent 后，这个 Prompt 会作为 SystemMessage
#          告诉 LLM 它有什么工具、应该按什么流程分析。
SENTIMENT_AGENT_PROMPT = """你是一位资深金融市场舆情分析师。你的任务是对给定企业的近期舆情做全面分析。

## 你的工具
- fetch_recent_news: 根据股票代码和企业名抓取近期新闻
- batch_score_news: 对新闻列表逐条做金融情感评分(看多/看空/中立)
- cluster_topics: 对评分后的新闻做主题聚类，发现热点话题

## 分析流程
1. 调用 fetch_recent_news 获取新闻
2. 调用 batch_score_news 逐条评分
3. 调用 cluster_topics 聚合并命名主题
4. 统计情感分布，识别核心舆情风险

## 输出要求
根据工具返回的数据，输出结构化的分析结果。重点关注:
- 负面情感占比是否异常
- 高危主题(监管/制裁/诉讼/关税)的出现频率
- 市场情绪的总体基调

用语简洁专业，避免主观臆断，所有判断需有新闻证据支撑。
"""


def _build_sentiment_prompt(user_input: SentimentInput) -> str:
    """
    构建传给 LLM 的用户指令文本。

    Phase 1 暂未使用——当前流程是代码控制工具链串行调用，
    LLM 只在最后一步润色摘要。

    Phase 2 切换 ReAct Agent 后，这个函数生成的文本会作为
    HumanMessage 传给 Agent，让 LLM 自主编排工具调用顺序。
    """
    return (
        f"请分析 {user_input.company_name}({user_input.symbol}) "
        f"近 {user_input.days} 天的市场舆情。\n"
        f"先用 fetch_recent_news 抓取新闻，再用 batch_score_news 逐条情感评分，"
        f"最后用 cluster_topics 聚类热点主题，并输出总结。"
    )


class SentimentAgent:
    """
    舆情分析 Agent。

    ══════════════════ 使用方式 ══════════════════
    from app.core.llm_factory import get_llm

    llm = get_llm("sentiment")
    agent = SentimentAgent(llm=llm)
    result = await agent.run(
        SentimentInput(symbol="300750", company_name="宁德时代", days=30)
    )
    # result 是 SentimentResult，可直接序列化或传给 RiskAgent
    ════════════════════════════════════════════════

    设计要点：
      - __init__ 接收 BaseChatModel：不写死模型类型，测试时注入 MockLLM
      - tools 是实例属性（不是类属性）：每个 Agent 实例独立持有
      - run() 是 async：支持真正的异步 LLM 调用（Phase 2）
      - prompt 是 @property：只读，防止被意外修改
    """

    def __init__(self, llm: BaseChatModel):
        """
        Args:
            llm: 聊天模型实例。生产环境用 LLMFactory.get_llm("sentiment")，
                 测试用 _MockLLM()。
        """
        self.llm = llm
        # 注册工具列表——Phase 2 切换 ReAct Agent 时
        # 传给 create_react_agent(llm, tools=self.tools)
        self.tools: list[BaseTool] = [fetch_recent_news, batch_score_news, cluster_topics]

    @property
    def prompt(self) -> str:
        """Agent 的 System Prompt，供外部读取（调试/日志用）"""
        return SENTIMENT_AGENT_PROMPT

    async def run(self, params: SentimentInput) -> SentimentResult:
        """
        执行舆情分析主流程。

        流程（5 步，严格串行）：
          Step 1: fetch_recent_news   → 抓取新闻列表
          Step 2: 空值检查            → 没新闻就直接返回（不崩）
          Step 3: batch_score_news    → 批量并行情感评分
          Step 4: 统计情感分布        → 计算正/负/中立数量
          Step 5: cluster_topics      → 热点主题聚类
          Step 6: LLM 生成摘要        → 自然语言总结

        为什么 Step 2 要做空值检查？
          如果某只股票近 30 天没有任何新闻（新股/冷门股），
          fetch_recent_news 返回空列表。后续的评分和聚类工具
          不会报错（它们能处理空列表），但 LLM 摘要需要知道「没有新闻」。
          提前返回带提示语的 SentimentResult 更清晰。
        """
        # ── Step 1: 新闻抓取 ──
        raw_news = fetch_recent_news.invoke({
            "symbol": params.symbol,
            "company_name": params.company_name,
            "days": params.days,
        })

        # ── Step 2: 空值保护 ──
        if not raw_news:
            return SentimentResult(
                symbol=params.symbol,
                company_name=params.company_name,
                searched_news_count=0,
                summary=f"未找到 {params.company_name} 近 {params.days} 天的相关新闻。",
            )

        # ── Step 3: 批量情感评分（asyncio.gather 并行） ──
        # 注意：batch_score_news 是 async @tool，run() 本身也是 async，
        # 必须用 await .ainvoke()，不能用同步 .invoke()（后者会在事件循环内崩掉）
        scored = await batch_score_news.ainvoke({"news_list": raw_news})

        # ── Step 4: 统计情感分布 ──
        dist: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        for sn in scored:
            # sn.sentiment.label 是 SentimentLabel 枚举，.value 返回 "positive"/"negative"/"neutral"
            dist[sn.sentiment.label.value] += 1

        # ── Step 5: 主题聚类 ──
        topics = cluster_topics.invoke({"scored_news": scored})

        # ── Step 6: LLM 生成舆情摘要 ──
        summary = await self._generate_summary(
            params=params,
            dist=dist,
            total=len(scored),
            topics=topics,
        )

        return SentimentResult(
            symbol=params.symbol,
            company_name=params.company_name,
            searched_news_count=len(raw_news),
            scored_news=scored,
            sentiment_distribution=dist,
            topics=topics,
            summary=summary,
        )

    async def _generate_summary(
        self,
        params: SentimentInput,
        dist: dict[str, int],
        total: int,
        topics: list,
    ) -> str:
        """
        使用 LLM 生成舆情分析摘要。

        输入给 LLM 的不是原始新闻文本（太长），而是统计摘要：
          - 抓了多少条
          - 正/负/中立各多少
          - 热点主题有哪些

        LLM 据此写 200 字以内小结，不需要逐条阅读新闻。

        Phase 2 优化：可改为 with_structured_output(SummarySchema)，
        让 LLM 返回结构化摘要（标题 + 要点列表），而非自由文本。
        """
        topic_desc = "\n".join(
            f"- {t.label}: {t.news_count}条, 关键词 [{', '.join(t.keywords[:5])}]"
            for t in topics
        )
        user_msg = (
            f"请根据以下数据，为 {params.company_name}({params.symbol}) "
            f"近 {params.days} 天的舆情做一段简洁的摘要(200字以内)：\n"
            f"共抓取 {total} 条新闻，情感分布: "
            f"看多{dist['positive']}/看空{dist['negative']}/中立{dist['neutral']}。\n"
            f"热点主题:\n{topic_desc or '暂无明确主题'}\n"
        )
        # await self.llm.ainvoke(user_msg) 发起真正的 LLM 调用
        resp = await self.llm.ainvoke(user_msg)
        # ChatModel 返回的 message 有 .content 属性
        return str(resp.content) if hasattr(resp, "content") else str(resp)
