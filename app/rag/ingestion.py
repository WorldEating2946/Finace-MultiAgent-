"""离线知识入库（Ingestion Pipeline）。

职责：
    file_path → load → fill company → split → embed → vector_store.add() → save()

对外访问：
    from app.rag import ingest

Phase 1：单文件入库，company 必填（一级过滤字段）。
入库后自动持久化到默认向量库目录（data/vector_store/），
下次启动 get_store(company_id=...) 直接加载，无需重复 ingest。
"""

from __future__ import annotations

from typing import Callable

from app.rag.document import DocumentChunk
from app.rag.embedding import EmbeddingModel, get_embedding_model
from app.rag.loaders import load_documents
from app.rag.splitter import split_documents
from app.rag.vectorstore import VectorRecord, VectorStore, get_store

ProgressCb = Callable[[int, int], None]  # (done, total)


def ingest(
    file_path: str,
    company: str,
    source_type: str = "",
    *,
    _model: EmbeddingModel | None = None,
    _store: VectorStore | None = None,
    progress_cb: ProgressCb | None = None,
) -> list[DocumentChunk]:
    """离线知识入库：加载 → 切片 → 向量化 → 写入向量库。

    Args:
        file_path:    源文件路径（.md / .txt / .pdf）。
        company:      所属企业（一级过滤字段，必填）。
        source_type:  文档语义类型（annual_report / research_report / policy / news），
                      PR #35 多源融合用；空串 = 不标注（默认，向后兼容）。

    Keyword Args:
        _model: 测试 seam —— 注入 EmbeddingModel。
        _store: 测试 seam —— 注入 VectorStore。

    Returns:
        list[DocumentChunk]: 入库后的 DocumentChunk 列表（已切分、已填 company）。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件类型不支持。

    Example:
        >>> chunks = ingest("docs/基金规则.md", company="测试公司", source_type="policy")
        >>> len(chunks)
        12
    """
    model = _model or get_embedding_model()
    store = _store or get_store(company_id=company)

    # 1. 加载
    docs = load_documents(file_path)

    # 2. 填充 company / source_type（写入 DocumentMetadata，
    #    splitter 提取到各 chunk.company / chunk.metadata["source_type"]）
    for doc in docs:
        doc.metadata.company = company
        doc.metadata.source_type = source_type

    # 3. 切片
    chunks = split_documents(docs)

    # 4. 向量化（有进度回调时按批嵌入并逐步上报，否则一次全量注入——零行为差异）
    texts = [c.text for c in chunks]
    vectors = _embed_texts(model, texts, progress_cb)

    # 5. 写入向量库（兼容新旧 store）
    if isinstance(store, VectorStore):
        # 新接口：VectorRecord 三合一写入（chunk_id + metadata + embedding 位置绑定）
        store.add(
            [
                VectorRecord.from_document_chunk(c, v)
                for c, v in zip(chunks, vectors)
            ]
        )
    else:
        # 旧接口（测试 seam 注入的旧 FAISSVectorStore）：两参数 add
        store.add(chunks, vectors)

    # 6. 持久化：本地 store（FAISS）save() 落盘；MilvusStore 服务端持久化、无 save()，
    #    getattr 守卫保证两种后端都能入库（勿改为硬调 store.save()）。
    save = getattr(store, "save", None)
    if callable(save):
        save()

    return chunks


_EMBED_BATCH_SIZE = 32  # 进度上报的嵌入批大小


def _embed_texts(model: EmbeddingModel, texts: list[str], progress_cb: ProgressCb | None) -> list[list[float]]:
    """嵌入文本；有进度回调时按 _EMBED_BATCH_SIZE 分批并逐批上报 (done, total)。"""
    if progress_cb is None or not texts:
        return model.embed(texts)

    vectors: list[list[float]] = []
    total = len(texts)
    for i in range(0, total, _EMBED_BATCH_SIZE):
        vectors.extend(model.embed(texts[i : i + _EMBED_BATCH_SIZE]))
        progress_cb(min(i + _EMBED_BATCH_SIZE, total), total)
    return vectors
