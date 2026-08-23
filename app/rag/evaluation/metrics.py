"""检索指标计算（PR #32）。

指标：
    - Recall@K：top-K 内期望章节命中的 query 占比；
    - Hit@K：    与 Recall@K 等价（binary relevance 下"命中"即"召回"）；
    - MRR：      第一个命中位置的倒数均值；
    - NDCG@K：   折扣累计增益——把 top-K 内后续命中也给分（MRR 只认第一个）。

NDCG 采用 binary gain（期望章节命中 = 1）。binary 下 NDCG 与 MRR 的差异：
MRR 只看第一个命中位置，NDCG@K 会奖励"命中了多个"的排序。

纯函数，不依赖 pipeline / LLM，可独立测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RetrievalMetrics:
    """检索质量指标汇总。"""

    recall_at_k: dict[int, float] = field(default_factory=dict)   # {5: 0.8, ...}
    hit_at_k: dict[int, float] = field(default_factory=dict)      # 与 recall 等价（binary）
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)     # {5: 0.42, ...}
    top1: float = 0.0                                             # Recall@1 别名（向后兼容）
    n: int = 0                                                    # query 数


def _ndcg_at(ranks: list[int], k: int) -> float:
    """单条 query 的 NDCG@K（binary gain：命中=1，未命中=0）。

    Args:
        ranks: 该 query 命中的 rank 列表（1-indexed，升序）。
        k:     截断深度。

    Returns:
        NDCG@K ∈ [0, 1]。无命中返回 0.0。
    """
    hits = [r for r in ranks if r <= k]
    if not hits:
        return 0.0
    dcg = sum(1.0 / math.log2(r + 1) for r in hits)
    # IDCG：把同样数量的命中放到前 n 位（理想排序）
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, len(hits) + 1))
    return dcg / idcg if idcg else 0.0


def compute_metrics(
    hit_ranks: list[list[int]],
    n_queries: int,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> RetrievalMetrics:
    """从每条 query 的命中 rank 列表聚合全部指标。

    Args:
        hit_ranks: 每条 query 的命中 rank 列表（1-indexed，升序，可为空）。
            同一 query 可在 top-K 内命中多个章节（各 chunk 命中都记录）。
        n_queries: 数据集 query 总数（含未命中的）。
        k_values:  要计算的截断深度集合。

    Returns:
        RetrievalMetrics。
    """
    if n_queries <= 0:
        return RetrievalMetrics()

    # 首命中 rank（MRR / Recall@K 用：只要任一章节命中即算该 query 命中）
    first_ranks = [ranks[0] if ranks else None for ranks in hit_ranks]

    recall_at_k: dict[int, float] = {}
    hit_at_k: dict[int, float] = {}
    ndcg_at_k: dict[int, float] = {}
    for k in k_values:
        recall_at_k[k] = sum(1 for r in first_ranks if r is not None and r <= k) / n_queries
        hit_at_k[k] = recall_at_k[k]
        ndcg_at_k[k] = sum(_ndcg_at(ranks, k) for ranks in hit_ranks) / n_queries

    mrr = sum(1.0 / r for r in first_ranks if r is not None) / n_queries

    return RetrievalMetrics(
        recall_at_k=recall_at_k,
        hit_at_k=hit_at_k,
        mrr=mrr,
        ndcg_at_k=ndcg_at_k,
        top1=recall_at_k.get(1, 0.0),
        n=n_queries,
    )
