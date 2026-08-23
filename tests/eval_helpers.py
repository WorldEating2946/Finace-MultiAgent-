"""真实年报评测共享逻辑（多公司）。

流程：质量门禁 → GPU + CrossEncoder → 存档复用 → 评测（expected_sections 命中任意）。

评测格式（evaluation/<company>_<year>.json）：
    [{"query": "...", "expected_sections": ["章节A", "章节B", ...]}, ...]
命中任意期望章节即 HIT（区分"评测数据歧义"与"模型问题"）。
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalResult:
    """单公司评测结果。

    PR #32：recall5/mrr/top1 保持向后兼容（旧调用方继续用）；
    新增 recall_at_k / hit_at_k / ndcg_at_k（全量指标，复用 app.rag.evaluation.metrics）。
    """

    recall5: float
    mrr: float
    top1: float
    n: int
    load_time: float = 0.0      # load_documents 耗时（损坏 PDF 含 OCR）
    query_time: float = 0.0     # 全部查询总耗时
    per_query: list[dict] = field(default_factory=list)
    # PR #32 新增：全量检索指标（Recall@1/5/10、Hit@K、NDCG@5/10）
    recall_at_k: dict = field(default_factory=dict)
    hit_at_k: dict = field(default_factory=dict)
    ndcg_at_k: dict = field(default_factory=dict)


def _load_eval(eval_path: str) -> list[dict]:
    """加载评测集：兼容旧格式（list of {query, expected_sections}）与新格式
    （{name, company, items: [...]}，PR #32 evaluation/datasets/）。"""
    data = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "items" in data:
        return [
            {"query": it["query"], "expected_sections": it.get("expected_sections", [])}
            for it in data["items"]
        ]
    return data


def run_company_eval(
    pdf_path: str,
    eval_path: str,
    company: str,
    pipeline_version: str,
) -> EvalResult:
    """执行公司年报评测，返回 EvalResult（含每题命中与耗时）。

    Args:
        pdf_path:         真实年报 PDF 路径。
        eval_path:        评测集 JSON 路径。
        company:          公司名（存档隔离与检索过滤）。
        pipeline_version: 管线版本（变化时强制重入库）。
    """
    import pytest

    from app.core.config import settings
    from app.rag import ingest, retrieve
    from app.rag.loaders import load_documents
    from app.rag.loaders.pdf_loader import PDFQualityError
    from app.rag.vectorstore import clear_store_cache, get_store

    if not Path(pdf_path).exists():
        pytest.skip(f"缺少真实 PDF: {pdf_path}")

    # 质量门禁（损坏 PDF 时 load_documents 内部走 OCR，耗时即 OCR 耗时）
    t0 = time.time()
    try:
        load_documents(str(pdf_path))
    except PDFQualityError:
        pytest.skip(f"PDF 文本层损坏且 OCR 失败（质量门禁阻断）: {pdf_path}")
    load_time = time.time() - t0

    # GPU（锦上添花）+ 真实 CrossEncoder
    try:
        import torch
        if torch.cuda.is_available():
            settings.rag_embedding_device = "cuda"
    except ImportError:
        pass
    from app.rag.reranker.cross_encoder import DEFAULT_RERANKER_PATH
    # 尊重 metadata 模式（PR #33 per-company 权重）；仅默认 dummy 时强制真实 CrossEncoder
    if settings.rag_reranker_model == "dummy" and Path(DEFAULT_RERANKER_PATH).is_dir():
        import app.rag.reranker as _rr
        settings.rag_reranker_model = DEFAULT_RERANKER_PATH
        _rr._default_reranker = None

    # 存档复用（管线版本一致则跳过重 embed）
    base = Path(tempfile.gettempdir()).resolve() / f"rag_eval_{company}"
    version_file = base / "pipeline.version"
    settings.rag_vector_store_path = str(base)
    # PR44.4：离线评测/回归门禁钉死 FAISS（临时目录重建语料），保证冻结基线可复现
    store = get_store(company_id=company, backend="faiss")
    cached = (
        version_file.exists()
        and version_file.read_text(encoding="utf-8").strip() == pipeline_version
        and store.count() > 0
    )
    if not cached:
        shutil.rmtree(base, ignore_errors=True)
        # 强制重置内存缓存：新旧工厂各自按 company 缓存单例并会 load() 旧索引，
        # 重建时必须清空两者，否则 ingest 会把新数据追加到旧索引（新旧混合）。
        from app.rag import vector_store as _vs

        _vs._default_stores.pop(company, None)  # 旧缓存（兼容残留调用方）
        clear_store_cache()  # 新缓存（PR44.2 get_store）
        settings.rag_vector_store_path = str(base)
        ingest(str(pdf_path), company=company)
        base.mkdir(parents=True, exist_ok=True)
        version_file.write_text(pipeline_version, encoding="utf-8")

    # 评测：expected_sections 命中任意 + 全量指标（PR #32，复用 metrics.py）
    data = _load_eval(eval_path)
    n = len(data)
    query_time = 0.0
    hit_ranks_all: list[list[int]] = []
    per_query: list[dict] = []
    for item in data:
        t_q = time.time()
        result = retrieve(item["query"], company=company, top_k=5)
        query_time += time.time() - t_q

        secs = [
            c.metadata.get("section", "") or c.metadata.get("chapter", "")
            for c in result.chunks
        ]
        expected = item["expected_sections"]
        # section 或 chapter 任一命中即 HIT：子标题（section）可能覆盖 outline 章节名
        # （如 ESG 区 "5.3 物流管理 > 5.3.3" 下的 chunk 仍属 "环境、社会及管治报告" 章）。
        # 记录全部命中 rank（NDCG 需要多个命中，MRR/Recall 只取第一个）。
        hit_ranks = [
            i + 1
            for i, c in enumerate(result.chunks)
            if any(
                exp in (c.metadata.get("section", "") or "")
                or exp in (c.metadata.get("chapter", "") or "")
                for exp in expected
            )
        ]
        hit_ranks_all.append(hit_ranks)

        top_chunk = result.chunks[0] if result.chunks else None
        per_query.append(
            {
                "query": item["query"],
                "expected_sections": expected,
                "hit_rank": hit_ranks[0] if hit_ranks else None,
                "hit_ranks": hit_ranks,
                "top1_page": top_chunk.page if top_chunk else None,
                "top1_section": secs[0] if secs else "",
                "top1_source": top_chunk.source if top_chunk else "",
            }
        )

    # 全量指标（NDCG@K / Hit@K / Recall@K / MRR / Top1）
    from app.rag.evaluation.metrics import compute_metrics

    m = compute_metrics(hit_ranks_all, n_queries=n)
    return EvalResult(
        recall5=m.recall_at_k.get(5, 0.0),
        mrr=m.mrr,
        top1=m.top1,
        n=n,
        load_time=load_time,
        query_time=query_time,
        per_query=per_query,
        recall_at_k=m.recall_at_k,
        hit_at_k=m.hit_at_k,
        ndcg_at_k=m.ndcg_at_k,
    )
