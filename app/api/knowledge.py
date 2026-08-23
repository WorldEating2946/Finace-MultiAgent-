"""知识库端点（检索 + 离线入库上传）。

对外暴露 RAG 检索能力；内部依赖 get_knowledge_retriever()（测试可注入 mock，
避免 API 测试加载真实 BGE/reranker 模型）。upload 走 app.rag.ingest 离线入库，
通过 SSE 流式推送入库进度（upload_progress：embedded/total），BGE-M3 嵌入在
单独线程执行（GPU 优先、无则 CPU），不阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import tempfile
import threading
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.response import ok
from app.models.user import User
from app.services.auth import get_current_user

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _sse(event: dict) -> str:
    """dict → SSE 帧（event: <type> / data: <json>）。"""
    etype = event.get("type", "message")
    return f"event: {etype}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def _ingested_aggs() -> dict[str, dict[str, int]]:
    """按向量后端汇总已入库公司 → {source_type: chunk 数}。"""
    if settings.rag_vector_backend == "milvus":
        from pymilvus import MilvusClient

        c = MilvusClient(uri=settings.milvus_uri, db_name=settings.milvus_db_name)
        if not c.has_collection(settings.milvus_collection_name):
            return {}
        rows = c.query(
            settings.milvus_collection_name,
            filter='chunk_id != ""',
            output_fields=["company_id", "metadata"],
            limit=16384,
        )
        aggs: dict[str, dict[str, int]] = {}
        for r in rows:
            comp = r.get("company_id") or ""
            if not comp:
                continue
            st = (r.get("metadata") or {}).get("source_type") or "unknown"
            aggs.setdefault(comp, {})
            aggs[comp][st] = aggs[comp].get(st, 0) + 1
        return aggs

    # faiss：data/vector_store/<company>/ 子目录 + index.faiss 存在算 1（近似）
    base = Path(settings.rag_vector_store_path)
    aggs: dict[str, dict[str, int]] = {}
    if base.exists():
        for d in base.iterdir():
            if d.is_dir():
                aggs.setdefault(d.name, {})["unknown"] = len(list(d.glob("index.faiss")))
    return aggs


@router.get("/companies")
def knowledge_companies(_: Annotated[User, Depends(get_current_user)]) -> dict:
    """已入库公司 + 文档类型清单（前端展示"哪些能查/不能查"）。"""
    aggs = _ingested_aggs()
    result = []
    for name in settings.rag_known_companies:
        d = aggs.get(name, {})
        result.append({
            "name": name,
            "ingested": bool(d),
            "source_types": [{"type": st, "count": ct} for st, ct in sorted(d.items())],
        })
    for name, d in sorted(aggs.items()):
        if name not in settings.rag_known_companies:
            result.append({
                "name": name,
                "ingested": True,
                "source_types": [{"type": st, "count": ct} for st, ct in sorted(d.items())],
            })
    return ok({"companies": result})


def get_knowledge_retriever():
    """依赖：RAG 检索函数（默认 retrieve；测试注入 mock）。"""
    from app.rag import retrieve

    return retrieve


@router.get("/search")
def knowledge_search(
    query: str = Query(..., min_length=1, description="分析问题"),
    company: str = Query(..., description="目标公司"),
    top_k: int = Query(5, ge=1, le=20, description="返回条数"),
    retriever=Depends(get_knowledge_retriever),
) -> dict:
    """语义检索：query + company → 命中 chunk 列表。"""
    result = retriever(query, company=company, top_k=top_k)
    chunks = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source": c.source,
            "page": c.page,
            "chapter": (c.metadata or {}).get("chapter", ""),
        }
        for c in result.chunks
    ]
    return ok({"query": query, "company": company, "chunks": chunks})


@router.post("/upload")
async def knowledge_upload(
    file: UploadFile = File(..., description="文档文件（.md/.txt/.pdf）"),
    company: str = Form(..., min_length=1, description="所属企业，如 宁德时代"),
    source_type: str = Form("", description="语义类型：annual_report/research_report/policy/news"),
    _: Annotated[User, Depends(get_current_user)] = None,
) -> StreamingResponse:
    """离线文档入库（SSE 流式进度）：upload_start → upload_progress×N → done/error。

    ingest 在独立线程执行（BGE-M3 嵌入，GPU 优先无则 CPU），进度经 callback
    推入队列，由 SSE 生成器逐帧产出，前端据此渲染进度条。
    """
    suffix = os.path.splitext(file.filename or "")[1] or ".txt"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as fh:
        fh.write(await file.read())

    async def _stream():
        loop = asyncio.get_running_loop()
        progress_q: queue.Queue = queue.Queue()
        held: dict = {}

        def on_progress(done: int, total: int) -> None:
            loop.call_soon_threadsafe(progress_q.put_nowait, {"embedded": done, "total": total})

        def run_ingest() -> None:
            from app.rag import ingest

            try:
                held["chunks"] = ingest(
                    tmp, company=company, source_type=source_type, progress_cb=on_progress
                )
            except Exception as exc:  # noqa: BLE001 —— 入库异常推 error 帧
                held["error"] = exc

        yield _sse({"type": "upload_start", "company": company})
        thread = threading.Thread(target=run_ingest, daemon=True)
        thread.start()

        while thread.is_alive():
            while not progress_q.empty():
                try:
                    yield _sse({"type": "upload_progress", **progress_q.get_nowait()})
                except queue.Empty:
                    break
            await asyncio.sleep(0.1)

        while not progress_q.empty():
            try:
                yield _sse({"type": "upload_progress", **progress_q.get_nowait()})
            except queue.Empty:
                break

        if "error" in held:
            yield _sse({"type": "error", "message": str(held["error"])})
        else:
            yield _sse({"type": "done", "chunk_count": len(held.get("chunks") or [])})

        if os.path.exists(tmp):
            os.remove(tmp)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
