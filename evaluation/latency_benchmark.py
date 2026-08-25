"""检索链路分阶段耗时基准（单次 query，在线查询阶段）。

运行：PYTHONPATH=. python evaluation/latency_benchmark.py
需要：宁德时代2025年报.pdf + 本地 BGE-M3 + bge-reranker-v2-m3

输出：Loader/Splitter/Ingest(离线) 与单次 query 的 Embedding/FAISS/BM25/RRF/CrossEncoder 耗时。

说明：CrossEncoder 是单次查询的性能瓶颈候选（用户每次 query 都跑 50 候选打分）；
Embedding 是离线 ingest 阶段，不计入在线 latency。
"""

from __future__ import annotations

import time
from pathlib import Path

from app.core.config import settings
from app.rag.dense_retriever import DenseRetriever
from app.rag.embedding import get_embedding_model
from app.rag.fusion import rrf_fuse
from app.rag.reranker.cross_encoder import DEFAULT_RERANKER_PATH, CrossEncoderReranker
from app.rag.sparse_retriever import SparseRetriever
from app.rag.vectorstore import get_store

PDF = Path(__file__).resolve().parent.parent / "宁德时代2025年报.pdf"
COMPANY = "宁德时代"
QUERY = "宁德时代2025年营业收入"
RETRIEVE_K = settings.rag_retrieve_top_k

settings.rag_embedding_device = "cuda"
settings.rag_reranker_model = DEFAULT_RERANKER_PATH


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms"


def _warm_query(query, model, store, sparse_r, reranker, fetch_k, company) -> None:
    """预热一次完整检索（触发模型加载/预热，不计入稳态计时）。"""
    model.embed([query])  # 触发 embedding 预热
    dense = DenseRetriever(model, store).search(query, top_k=fetch_k, company=company)
    sparse = sparse_r.search(query, top_k=fetch_k, company=company)
    fused = rrf_fuse(dense, sparse)[:fetch_k]
    reranker.rerank(query, [c for c, _ in fused])


def main() -> None:
    # ── 离线：load / split / ingest（一次性）────────────────────
    t0 = time.time()
    from app.rag.loaders import load_documents
    from app.rag.splitter import split_documents

    docs = load_documents(str(PDF))
    t_load = time.time() - t0
    t0 = time.time()
    chunks = split_documents(docs, chunk_size=512, chunk_overlap=100)
    t_split = time.time() - t0
    print(f"[离线] Loader={_ms(t_load)}  Splitter={_ms(t_split)}  docs={len(docs)} chunks={len(chunks)}")

    store = get_store(company_id=COMPANY)
    if store.count() == 0:
        t0 = time.time()
        from app.rag.ingestion import ingest

        ingest(str(PDF), company=COMPANY)
        print(f"[离线] Ingest(含 Embedding)={_ms(time.time() - t0)}")
    else:
        print("[离线] 存档复用，跳过 Ingest")

    # ── 在线：单次 query 分阶段 ─────────────────────────────────
    model = get_embedding_model()
    reranker = CrossEncoderReranker()

    # BM25 语料一次构建（在线稳态下已缓存）；新接口 all_chunks() 返回
    # VectorRecord → 桥接为 DocumentChunk（SparseRetriever.build() 消费后者）
    t0 = time.time()
    sparse_r = SparseRetriever().build(
        [r.to_document_chunk(COMPANY) for r in store.all_chunks()]
    )
    print(f"[在线·一次] BM25 语料构建={_ms(time.time() - t0)}")

    # warmup：首次调用含模型加载/预热，不计入稳态
    _warm_query(QUERY, model, store, sparse_r, reranker, RETRIEVE_K, COMPANY)

    # 稳态：第二次 query 分阶段计时
    t0 = time.time()
    model.embed([QUERY])
    t_embed = time.time() - t0
    t0 = time.time()
    dense = DenseRetriever(model, store).search(QUERY, top_k=RETRIEVE_K, company=COMPANY)
    t_faiss = time.time() - t0
    t0 = time.time()
    sparse = sparse_r.search(QUERY, top_k=RETRIEVE_K, company=COMPANY)
    t_bm25 = time.time() - t0
    t0 = time.time()
    fused = rrf_fuse(dense, sparse)[:RETRIEVE_K]
    t_rrf = time.time() - t0
    t0 = time.time()
    reranked = reranker.rerank(QUERY, [c for c, _ in fused])
    t_rerank = time.time() - t0
    total = t_embed + t_faiss + t_bm25 + t_rrf + t_rerank

    print(f"\n[在线] 单次 query latency（Retrieve {RETRIEVE_K} → Rerank {len(reranked)}）")
    print(f"  Embedding     {_ms(t_embed)}")
    print(f"  FAISS(Dense)  {_ms(t_faiss)}")
    print(f"  BM25(Sparse)  {_ms(t_bm25)}")
    print(f"  RRF           {_ms(t_rrf)}")
    print(f"  CrossEncoder  {_ms(t_rerank)}   ← 在线瓶颈候选")
    print("  ────────────────────────────")
    print(f"  总 Retrieval  {_ms(total)}")


if __name__ == "__main__":
    main()
