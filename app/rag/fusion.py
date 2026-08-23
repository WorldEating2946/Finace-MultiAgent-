"""检索结果融合（Reciprocal Rank Fusion）。

Dense 与 Sparse 双路各有优劣：Dense 语义优先，Sparse 关键词优先。
RRF 基于排序位置融合（不依赖分数尺度可比性）：

    score(chunk) = Σ 1/(k + rank_route)

例：
    Dense：A rank1, B rank2, C rank5
    BM25：B rank1, D rank2, A rank4
    融合（k=60）：B > A > D > C
"""

from __future__ import annotations

from app.rag.document import DocumentChunk

_RRF_K_DEFAULT = 60


def rrf_fuse(
    *candidate_lists: list[tuple[DocumentChunk, float]],
    k: int = _RRF_K_DEFAULT,
) -> list[tuple[DocumentChunk, float]]:
    """Reciprocal Rank Fusion：score = Σ 1/(k + rank)，归一化到 [0, 1]。

    Args:
        candidate_lists: 一个或多个已按相关度降序的 (chunk, score) 列表。
            双路调用 rrf_fuse(dense, sparse)；multi-query 调用 rrf_fuse(*q_results)。

    Returns:
        融合后按分数降序的 (chunk, 归一化分数) 列表。
    """
    agg: dict[str, list] = {}  # chunk_id -> [score, chunk]
    for candidates in candidate_lists:
        for rank, (chunk, _) in enumerate(candidates, 1):
            entry = agg.setdefault(chunk.chunk_id, [0.0, chunk])
            entry[0] += 1.0 / (k + rank)

    if not agg:
        return []
    max_score = max(entry[0] for entry in agg.values())
    ranked = sorted(agg.values(), key=lambda e: e[0], reverse=True)
    return [(entry[1], entry[0] / max_score) for entry in ranked]
