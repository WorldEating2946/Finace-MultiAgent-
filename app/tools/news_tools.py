"""
新闻实时抓取工具

封装外部新闻源调用，为 Sentiment Agent 提供新闻数据。

════════════════════ 为什么封装成 @tool？ ════════════════════════
LangChain @tool 是所有外部能力的统一封装格式。
好处：
  1. LLM 可以通过 Function Calling 自动决定何时调用（Phase 2）
  2. 输入/输出类型明确，自动生成 JSON Schema
  3. Phase 1 返回 mock 数据，Phase 2 替换为真实 API——接口不变

当前 Phase 1 返回占位数据，保证链路通畅可测。
Phase 2 替换 fetch_recent_news 的实现为真实 HTTP 请求即可。
═══════════════════════════════════════════════════════════════
"""

import logging
from datetime import datetime, timedelta

from langchain_core.tools import tool

from ..models.sentiment_risk_models import NewsItem

logger = logging.getLogger(__name__)

# ── 新闻源配置 ──────────────────────────────────────────────
# 真实新闻优先用 akshare stock_news_em（东方财富）；以下站点 URL 留作扩展 seed。
_NEWS_SOURCES: list[dict] = [
    {"name": "东方财富", "base_url": "https://so.eastmoney.com/news/s"},
    {"name": "巨潮资讯", "base_url": "http://www.cninfo.com.cn/new/commonUrl"},
    {"name": "财联社",   "base_url": "https://www.cls.cn/searchPage"},
]


@tool
def fetch_recent_news(symbol: str, company_name: str, days: int = 30) -> list[NewsItem]:
    """
    根据股票代码和企业名抓取近期新闻。

    这是 Sentiment Agent 的第一个工具，后续情感评分和主题聚类都依赖它的输出。
    数据流：symbol + company_name → HTTP 请求新闻 API → list[NewsItem]

    Args:
        symbol:      股票代码，如 '300750'
        company_name: 企业名称，如 '宁德时代'
        days:        回溯天数，默认 30，范围 [1, 365]

    Returns:
        结构化新闻列表，每条包含标题、来源、链接、发布时间、摘要
    """
    # 真实新闻：akshare stock_news_em（东方财富），失败/空回退占位
    try:
        import akshare as ak

        df = ak.stock_news_em(symbol=symbol)
    except Exception as exc:  # noqa: BLE001 —— akshare/网络异常降级
        logger.warning("akshare 新闻获取失败，降级占位: %s", exc)
        return _placeholder_news(symbol, company_name, days)

    if df is None or df.empty:
        logger.warning("akshare 新闻为空，降级占位: %s", symbol)
        return _placeholder_news(symbol, company_name, days)

    cutoff = datetime.now() - timedelta(days=days)
    items: list[NewsItem] = []
    for _, row in df.iterrows():
        title = str(row.get("新闻标题", "")).strip()
        if not title:
            continue
        published_raw = row.get("发布时间", None)
        published_at = None
        if published_raw:
            try:
                published_at = datetime.fromisoformat(str(published_raw))
            except ValueError:
                published_at = None
        if published_at and published_at < cutoff:
            continue
        items.append(
            NewsItem(
                title=title,
                source=str(row.get("文章来源", "东方财富")),
                url=str(row.get("新闻链接", "")),
                published_at=published_at or datetime.now(),
                summary=str(row.get("新闻内容", "")).strip()[:500],
            )
        )
        if len(items) >= 20:
            break
    return items or _placeholder_news(symbol, company_name, days)


def _placeholder_news(symbol: str, company_name: str, days: int) -> list[NewsItem]:
    """akshare 不可用时的占位返回（3 来源 × 3 轮 = 9 条）。"""
    return [
        NewsItem(
            title=f"{company_name}({symbol}) 近期动态{i+1}",
            source=src["name"],
            url=f"{src['base_url']}?keyword={symbol}",
            published_at=datetime.now() - timedelta(days=i),  # noqa: DTZ005 —— 占位相对时间戳
            summary=f"关于{company_name}的新闻摘要 #{i+1}，实际接入API后替换为真实内容。",
        )
        for i, src in enumerate(_NEWS_SOURCES * 3)  # 3 来源 × 3 轮 = 9 条
    ]


def get_news_tools() -> list:
    """
    返回 Sentiment Agent 可用的新闻工具列表。

    LangChain Agent 需要以列表形式注册工具：
      agent = SentimentAgent(llm=llm)
      agent.tools = [fetch_recent_news, batch_score_news, cluster_topics]

    get_news_tools() 只是便利函数，方便统一收集所有工具。
    """
    return [fetch_recent_news]
