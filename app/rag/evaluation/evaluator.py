"""评测执行器（PR #32）。

RagEvaluator：逐条 query 检索 → 计算全量指标（Recall@K / Hit@K / MRR / NDCG）。
PipelineBenchmark：分阶段 latency 报告（rewrite → retrieval → rerank）。

与 tests/eval_helpers.run_company_eval 的分工：
    - run_company_eval：负责 store 构建（质量门禁 + 存档复用 + ingest）+ 评测；
    - RagEvaluator：纯测量（假设 store 已就绪），供回归脚本 / 测试 / 后续 PR 复用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.rag.evaluation.dataset import EvaluationDataset
from app.rag.evaluation.metrics import RetrievalMetrics, compute_metrics


class RagEvaluator:
    """执行评测数据集 → 返回指标 + per-query 明细。假设向量库已就绪。"""

    def __init__(
        self,
        company: str,
        rewriter=None,
        top_k: int = 10,
        *,
        _model=None,
        _store=None,
    ) -> None:
        self._company = company
        self._rewriter = rewriter
        self._top_k = top_k
        self._model = _model      # 测试 seam —— 注入 EmbeddingModel
        self._store = _store      # 测试 seam —— 注入 VectorStore

    def evaluate(
        self,
        dataset: EvaluationDataset,
    ) -> tuple[RetrievalMetrics, list[dict]]:
        """逐条 query 检索并计算指标。

        Returns:
            (RetrievalMetrics, per_query_details)。
            per_query_details: [{id, query, hit_ranks, top1_section, top1_source}]。
        """
        from app.rag import retrieve

        hit_ranks: list[list[int]] = []
        details: list[dict] = []
        for item in dataset.items:
            result = retrieve(
                item.query,
                company=item.company or dataset.company,
                top_k=self._top_k,
                _rewriter=self._rewriter,
                _model=self._model,
                _store=self._store,
            )
            ranks: list[int] = []
            for i, c in enumerate(result.chunks, 1):
                sec = c.metadata.get("section") or ""
                ch = c.metadata.get("chapter") or ""
                if any(e in sec or e in ch for e in item.expected_sections):
                    ranks.append(i)
            hit_ranks.append(ranks)

            top = result.chunks[0] if result.chunks else None
            details.append(
                {
                    "id": item.id,
                    "query": item.query,
                    "hit_ranks": ranks,
                    "top1_section": (
                        (top.metadata.get("section") or top.metadata.get("chapter") or "")
                        if top
                        else ""
                    ),
                    "top1_source": top.source if top else "",
                }
            )

        metrics = compute_metrics(hit_ranks, n_queries=len(dataset.items))
        return metrics, details


class PipelineBenchmark:
    """分阶段 latency 基准（rewrite → retrieval → rerank），单进程稳态。"""

    @dataclass
    class Result:
        n_queries: int
        avg_latency_ms: dict[str, float] = field(default_factory=dict)  # rewrite/retrieval/rerank/total
        per_query: list[dict] = field(default_factory=list)

        def format_report(self) -> str:
            lines = [f"Queries: {self.n_queries}", "", "Latency:", ""]
            lines.append(f"  Rewrite:     {self.avg_latency_ms.get('rewrite', 0):.0f}ms")
            lines.append(f"  Retrieval:   {self.avg_latency_ms.get('retrieval', 0):.0f}ms")
            lines.append(f"  Rerank:      {self.avg_latency_ms.get('rerank', 0):.0f}ms")
            lines.append(f"  Total:       {self.avg_latency_ms.get('total', 0):.0f}ms")
            return "\n".join(lines)

    def run(
        self,
        dataset: EvaluationDataset,
        *,
        warmup: bool = True,
        fetch_k: int = 20,
    ) -> Result:
        """逐条 query 跑完整 pipeline，记录每阶段耗时。

        Args:
            dataset: 评测数据集（company 用于向量库过滤）。
            warmup:  是否先跑 1 次触发模型加载/预热（不计入稳态）。
            fetch_k: 单路召回候选数（与 retriever._FETCH_K 一致）。
        """
        from app.core.config import settings
        from app.rag.dense_retriever import DenseRetriever
        from app.rag.embedding import get_embedding_model
        from app.rag.fusion import rrf_fuse
        from app.rag.query import get_query_rewriter
        from app.rag.reranker import get_reranker
        from app.rag.sparse_retriever import SparseRetriever
        from app.rag.vectorstore import get_store

        company = dataset.company
        model = get_embedding_model()
        # PR44.4：离线评测钉死 FAISS —— 冻结基线严格可复现，不随生产配置（milvus）漂移
        store = get_store(company_id=company, backend="faiss")
        reranker = get_reranker()
        rewriter = get_query_rewriter()
        # BM25 语料一次构建（在线稳态下已缓存）；新接口 all_chunks() 返回
        # VectorRecord → 桥接为 DocumentChunk（SparseRetriever.build() 仍消费后者）
        sparse_r = SparseRetriever().build(
            [r.to_document_chunk(company) for r in store.all_chunks()]
        )

        def _recall(sq: str) -> list:
            dense = DenseRetriever(model, store).search(sq, top_k=fetch_k, company=company)
            if settings.rag_hybrid and dense:
                try:
                    sparse = sparse_r.search(sq, top_k=fetch_k, company=company)
                    return rrf_fuse(dense, sparse)[:fetch_k]
                except ImportError:
                    pass
            return rrf_fuse(dense, [])[:fetch_k]

        def _query_flow(q: str) -> tuple[float, float, float]:
            """返回 (rewrite_ms, retrieval_ms, rerank_ms)。"""
            t0 = time.time()
            sub_queries = rewriter.rewrite(q)
            t_rewrite = (time.time() - t0) * 1000

            t0 = time.time()
            all_fused = [_recall(sq) for sq in sub_queries]
            all_fused = [f for f in all_fused if f]
            fused = rrf_fuse(*all_fused)[:fetch_k] if all_fused else []
            t_retrieval = (time.time() - t0) * 1000

            t0 = time.time()
            reranker.rerank(q, [c for c, _ in fused])
            t_rerank = (time.time() - t0) * 1000

            return t_rewrite, t_retrieval, t_rerank

        queries = [item.query for item in dataset.items]
        if warmup and queries:
            _query_flow(queries[0])  # 触发模型加载/预热，不计入稳态

        per_query: list[dict] = []
        sums = {"rewrite": 0.0, "retrieval": 0.0, "rerank": 0.0}
        for q in queries:
            rw, rt, rr = _query_flow(q)
            total = rw + rt + rr
            sums["rewrite"] += rw
            sums["retrieval"] += rt
            sums["rerank"] += rr
            per_query.append({"query": q, "rewrite_ms": rw, "retrieval_ms": rt, "rerank_ms": rr, "total_ms": total})

        n = len(queries)
        avg = {k: v / n if n else 0.0 for k, v in sums.items()}
        avg["total"] = avg["rewrite"] + avg["retrieval"] + avg["rerank"]

        return PipelineBenchmark.Result(n_queries=n, avg_latency_ms=avg, per_query=per_query)
