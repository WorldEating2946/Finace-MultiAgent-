"""
金融情感分析工具

封装 FinBERT 情感评分 和 BERTopic 主题聚类，供 Sentiment Agent 调用。

════════════════════ 数据流 ════════════════════════
fetch_recent_news（抓新闻）
    ↓
score_sentiment（逐条 FinBERT 评分）
    ↓
batch_score_news（批量并行评分，asyncio.gather 加速）
    ↓
cluster_topics（BERTopic 主题聚类）
    ↓
SentimentResult（汇总，交给 Risk Agent / Report Agent）

Phase 1 返回占位评分。Phase 2 接入真实模型：
  - FinBERT: ProsusAI/finbert，HuggingFace transformers 本地推理
  - BERTopic: 主题聚类模型，自动发现新闻热点
═══════════════════════════════════════════════════════════
"""

import asyncio
import logging
import os
import threading

from langchain_core.tools import tool

from ..models.sentiment_risk_models import (
    NewsItem,
    ScoredNews,
    SentimentLabel,
    SentimentScore,
    TopicCluster,
)

logger = logging.getLogger(__name__)

# 情感模型名（中文三分类：positive/neutral/negative）
# ProsusAI/finbert 是英文模型，中文全 neutral；改用它支持中文金融情绪三分类。
_FINBERT_MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

# FinBERT 单例（惰性加载 + 线程锁，避免重复下载/加载）
_pipe = None
_pipe_lock = threading.Lock()

_LABEL_ALIAS = {
    "positive": "positive", "POSITIVE": "positive",
    "negative": "negative", "NEGATIVE": "negative",
    "neutral": "neutral", "NEUTRAL": "neutral",
}


def _flatten_label_dicts(seq) -> list[dict]:
    """递归摊平 FinBERT pipeline 输出为 [{label, score}, ...]（兼容单条/批量嵌套）。"""
    out: list[dict] = []
    for x in seq:
        if isinstance(x, dict):
            out.append(x)
        elif isinstance(x, list):
            out.extend(_flatten_label_dicts(x))
    return out


def _finbert_cached() -> bool:
    """FinBERT 是否已本地缓存（HuggingFace 不可达时避免联网下载导致的挂起/超时）。"""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo = "models--" + _FINBERT_MODEL_NAME.replace("/", "--")
        return os.path.isdir(os.path.join(HF_HUB_CACHE, repo))
    except Exception:  # noqa: BLE001 —— 缓存目录不可用则回退尝试加载
        return True


def _get_finbert():
    """惰性加载 FinBERT pipeline；失败/未缓存返回 None（评分降级 NEUTRAL）。

    关键：HF 不可达时**不联网尝试**（本地无缓存 → 直接返回 None），
    避免 sentiment 端点每次卡 50s 等下载超时。
    """
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                if not _finbert_cached():
                    logger.warning("FinBERT 未本地缓存（HuggingFace 不可达），评分降级 NEUTRAL")
                    return None
                try:
                    from transformers import pipeline

                    _pipe = pipeline(
                        "sentiment-analysis",
                        model=_FINBERT_MODEL_NAME,
                        top_k=None,
                        truncation=True,
                        max_length=512,
                    )
                except Exception:  # noqa: BLE001 —— 加载失败 → 降级
                    logger.warning("FinBERT 加载失败，评分降级", exc_info=True)
                    _pipe = None
    return _pipe


@tool
def score_sentiment(news_text: str) -> SentimentScore:
    """
    使用 FinBERT 对单条新闻文本做金融情感评分。

    输入一条新闻的标题+摘要拼接文本，输出一个 SentimentScore：
      - label：POSITIVE（看多）/ NEGATIVE（看空）/ NEUTRAL（中立）
      - confidence：[0,1] 置信度
      - explanation：判断依据简述

    Phase 1：返回占位值（NEUTRAL, 0.5）。
    Phase 2：替换为 FinBERT 本地推理。

    Args:
        news_text: 新闻标题 + 摘要拼接文本，如 '宁德时代发布新电池技术...'

    Returns:
        SentimentScore: 情感标签 + 置信度 + 判断依据
    """
    # 真实 FinBERT 推理；不可用/异常 → 降级 NEUTRAL/0.5（保证链路不崩）
    pipe = _get_finbert()
    if pipe is None:
        return SentimentScore(
            label=SentimentLabel.NEUTRAL,
            confidence=0.5,
            explanation="[降级] FinBERT 不可用",
        )
    try:
        # top_k=None 时 pipeline 可能返回 [{label,score}...] 或 [[{label,score}...]]（按输入条目嵌套）
        result = pipe(news_text[:512]) or []
        labels = _flatten_label_dicts(result)
        top = max(labels, key=lambda x: x.get("score", 0)) if labels else None
        if top is None:
            raise ValueError("finbert 无输出")
        label = _LABEL_ALIAS.get(str(top.get("label", "")), "neutral")
        conf = float(top.get("score", 0.5))
        return SentimentScore(
            label=SentimentLabel(label),
            confidence=conf,
            explanation=f"FinBERT: {label}={conf:.2f}",
        )
    except Exception:  # noqa: BLE001 —— 评分异常降级
        logger.warning("FinBERT 评分异常，降级", exc_info=True)
        return SentimentScore(
            label=SentimentLabel.NEUTRAL,
            confidence=0.5,
            explanation="[降级] FinBERT 评分异常",
        )


@tool
async def batch_score_news(news_list: list[NewsItem]) -> list[ScoredNews]:
    """
    批量情感评分——asyncio.gather 并行调用 FinBERT。

    ══════════════════ 性能关键点 ══════════════════
    原来：for news in news_list → score_sentiment.invoke(...)
         每条 5-10 秒（如果接真实 LLM），9 条 = 45-90 秒，串行累加。

    现在：asyncio.gather(*tasks)
         9 条同时发出去，总耗时 ≈ 最慢的那条（5-10 秒），提速约 9 倍。

    实现方式：
      1. 内层 _score_one(news) 是 async 函数，每条独立调用 score_sentiment.ainvoke()
      2. 外层用 asyncio.gather 把 9 个协程一起跑，等全部完成后返回
      3. gather 结果顺序与 tasks 顺序严格一致
    ═════════════════════════════════════════════════

    Args:
        news_list: 新闻列表

    Returns:
        附带情感评分的新闻列表，顺序与输入一致
    """
    async def _score_one(news: NewsItem) -> ScoredNews:
        """单条新闻评分——内层函数，每条独立，某条失败不影响其他"""
        text = f"{news.title} {news.summary}"
        sentiment = await score_sentiment.ainvoke({"news_text": text})
        return ScoredNews(news=news, sentiment=sentiment)

    # 先创建所有协程（此时还没执行，只是"准备好"）
    tasks = [_score_one(news) for news in news_list]
    # 再并发执行——await asyncio.gather 等全部完成
    results = await asyncio.gather(*tasks)
    return list(results)


@tool
def cluster_topics(scored_news: list[ScoredNews], n_topics: int = 5) -> list[TopicCluster]:
    """
    使用 BERTopic 对带情感的新闻做主题聚类。

    输入带情感评分的新闻列表，自动发现热点话题——
    如'欧美关税政策'、'固态电池突破'、'高管变更'。

    原理（Phase 2）：BERTopic 将每条新闻转成向量，用 HDBSCAN 聚类，
    自动为每个簇生成关键词和标签。

    Phase 1：返回占位聚类（所有新闻归为一个主题）。
    Phase 2：接入 bertopic 库，替换返回逻辑。

    Args:
        scored_news: 带情感评分的新闻列表
        n_topics:   最大主题数，默认 5

    Returns:
        主题聚类列表，每个包含标签、关键词、新闻数量、代表性标题
    """
    # 轻量关键词主题聚类（BERTopic 未装，用 jieba 关键词分组，确定性、免重依赖）
    if not scored_news:
        return []

    import jieba
    from collections import Counter, defaultdict

    stop = set("的 了 在 是 与 和 及 或 称 就 将 中 为 等 更 这 那 也 对 从 会 可 以 一个 公司 股票 市场".split())
    corpus: list[set[str]] = []
    for sn in scored_news:
        text = f"{sn.news.title} {sn.news.summary}"
        tokens = [w for w in jieba.lcut(text) if len(w) >= 2 and w not in stop]
        corpus.append(set(tokens))

    freq: Counter[str] = Counter()
    for toks in corpus:
        freq.update(toks)

    groups: dict[str, list[int]] = defaultdict(list)
    for i, toks in enumerate(corpus):
        if toks:
            top_kw = max(toks, key=lambda w: freq[w])
            groups[top_kw].append(i)

    clusters: list[TopicCluster] = []
    tid = 0
    for kw, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:n_topics]:
        clusters.append(
            TopicCluster(
                topic_id=tid,
                label=f"主题：{kw}",
                keywords=[kw],
                news_count=len(idxs),
                representative_news=[scored_news[i].news.title for i in idxs[:3]],
            )
        )
        tid += 1
    return clusters


def get_sentiment_tools() -> list:
    """
    返回 Sentiment Agent 可用的情感工具列表。

    三个工具按调用顺序排列：
      1. score_sentiment  —— 单条评分
      2. batch_score_news —— 批量并行评分（内部调用 score_sentiment）
      3. cluster_topics    —— 主题聚类
    """
    return [score_sentiment, batch_score_news, cluster_topics]
