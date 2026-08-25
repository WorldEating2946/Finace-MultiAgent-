"""PR44.2 FAISS Adapter 测试（第一个完全符合 VectorStore Contract 的生产 Adapter）。

测试策略：
    - 真实 FAISSStore（包装旧 FAISSVectorStore，无 Mock），验证"生产适配器"
      在 add/search/delete/update/count/compact 全生命周期下符合契约；
    - 向量由 DummyEmbeddingModel 生成（MD5 确定性——测试依赖过滤/生命周期逻辑
      而非语义排序，top_k 取大保证全量召回再断言过滤结果）；
    - 每个测试独立实例，工厂类测试用 clear_store_cache() 隔离单例；
    - 迁移验证：DenseRetriever / ingest 与"新 store"协同工作，旧 get_vector_store 仍可用。
"""


from app.rag.document import DocumentChunk
from app.rag.embedding import DummyEmbeddingModel
from app.rag.vectorstore import (
    FAISSStore,
    SearchResult,
    VectorRecord,
    clear_store_cache,
    get_store,
)

DIM = 128


# ── 测试辅助 ──────────────────────────────────────────────────────

def _make_record(
    text: str,
    chunk_id: str,
    *,
    company_id: str = "xiaomi",
    year: int | None = None,
    source_type: str | None = None,
    **extra,
) -> VectorRecord:
    """创建测试用 VectorRecord（metadata 含 enterprise 过滤字段）。"""
    meta = {"company_id": company_id}
    if year is not None:
        meta["year"] = year
    if source_type is not None:
        meta["source_type"] = source_type
    meta.update(extra)
    return VectorRecord(
        chunk_id=chunk_id,
        text=text,
        embedding=DummyEmbeddingModel(dim=DIM).embed([text])[0],
        metadata=meta,
    )


def _make_store(company: str = "xiaomi", dir_path=None) -> FAISSStore:
    """创建独立的 FAISSStore 实例（非全局单例）。"""
    return FAISSStore(dim=DIM, company=company, dir_path=dir_path)


def _query(text: str) -> list[float]:
    return DummyEmbeddingModel(dim=DIM).embed([text])[0]


# ── A. 契约检索（新接口 SearchResult + filters） ─────────────────

def test_search_returns_search_result():
    """新接口 search 返回 SearchResult 列表（非裸元组）。"""
    store = _make_store()
    store.add([_make_record(f"小米年报 {i}", f"c{i}") for i in range(3)])

    results = store.search(_query("小米年报"), top_k=10)

    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_search_filters_company_id():
    """filters={"company_id": "xiaomi"} 正确过滤（公司隔离契约）。"""
    store = _make_store(company="xiaomi")
    store.add([_make_record("小米内容", "c0", company_id="xiaomi")])

    hit = store.search(_query("小米"), top_k=10, filters={"company_id": "xiaomi"})
    miss = store.search(_query("小米"), top_k=10, filters={"company_id": "catl"})

    assert {r.chunk_id for r in hit} == {"c0"}
    assert miss == []


def test_search_result_to_document_chunk():
    """SearchResult.to_document_chunk() 构造 DocumentChunk（桥接 retriever 管线）。"""
    sr = SearchResult(
        chunk_id="c1",
        text="小米 2025 年报营收",
        score=0.87,
        metadata={"company_id": "xiaomi", "source_type": "annual_report", "page": 3},
    )

    chunk = sr.to_document_chunk()

    assert isinstance(chunk, DocumentChunk)
    assert chunk.chunk_id == "c1"
    assert chunk.text == "小米 2025 年报营收"
    assert chunk.company == "xiaomi"
    assert chunk.page == 3
    assert chunk.metadata["source_type"] == "annual_report"
    # 显式 company 覆盖 metadata
    assert sr.to_document_chunk("catl").company == "catl"


# ── B. 生命周期：delete / update ─────────────────────────────────

def test_delete_then_search_absent():
    """insert → search 命中 → delete → search 不命中。"""
    store = _make_store()
    store.add([_make_record("保留", "keep"), _make_record("删除", "drop")])

    assert {r.chunk_id for r in store.search(_query("内容"), top_k=10)} == {"keep", "drop"}

    removed = store.delete(["drop"])

    assert removed == 1
    assert {r.chunk_id for r in store.search(_query("内容"), top_k=10)} == {"keep"}
    assert store.count() == 1


def test_update_then_search_returns_new():
    """insert old → update → search 返回 new chunk（旧内容不再出现）。"""
    store = _make_store()
    store.add([_make_record("旧版本内容", "c1", year=2024)])

    updated = store.update(_make_record("新版本内容", "c1", year=2025))

    assert updated is True
    assert store.count() == 1  # 旧版本逻辑删除，不重复计数
    results = store.search(_query("版本"), top_k=10)
    assert len(results) == 1
    assert results[0].text == "新版本内容"
    assert results[0].metadata["year"] == 2025


# ── C. compact() 物理重建 ────────────────────────────────────────

def test_compact_physically_removes_deleted(tmp_path):
    """add + delete → compact() → 索引与 metadata 物理移除已删 chunk。"""
    store = FAISSStore(dim=DIM, company="xiaomi", dir_path=tmp_path)
    store.add(
        [
            _make_record(f"小米年报 {i}", f"c{i}") for i in range(3)
        ]
    )
    store.delete(["c1"])
    store.save()  # compact 前的存档（含 deleted 标记）
    assert store.validate_integrity()["deleted"] == 1

    removed = store.compact()  # 内部自动 save()

    assert removed == 1
    integrity = store.validate_integrity()
    assert integrity["ntotal"] == 2
    assert integrity["metadata_count"] == 2
    assert integrity["deleted"] == 0  # 逻辑删除标记随物理移除归零
    assert integrity["consistent"] is True

    # 重新加载：物理删除已落盘（compact 内部 save 写入）
    loaded = FAISSStore(dim=DIM, company="xiaomi", dir_path=tmp_path)
    loaded.load()
    assert loaded.validate_integrity()["ntotal"] == 2
    assert loaded.validate_integrity()["deleted"] == 0
    assert loaded.count() == 2


def test_compact_preserves_active_chunks():
    """add 3 + delete 1 → compact → 活跃 2 条可检索、已删 chunk 物理消失。"""
    store = _make_store()
    store.add([_make_record(f"小米年报 {i}", f"c{i}") for i in range(3)])
    store.delete(["c1"])

    removed = store.compact()

    assert removed == 1
    assert store.count() == 2
    results = store.search(_query("年报"), top_k=10)
    assert {r.chunk_id for r in results} == {"c0", "c2"}
    assert store.vector_of("c1") is None  # 已物理移除
    assert store.vector_of("c0") is not None  # 活跃 chunk 向量仍在（reconstruct 正确）


def test_compact_noop_when_no_deleted():
    """无 deleted 标记时 compact() 返回 0，数据不变。"""
    store = _make_store()
    store.add([_make_record(f"内容 {i}", f"c{i}") for i in range(3)])

    removed = store.compact()

    assert removed == 0
    assert store.count() == 3
    assert {r.chunk_id for r in store.search(_query("内容"), top_k=10)} == {"c0", "c1", "c2"}


def test_compact_on_empty_store_returns_zero():
    """空库 compact() 返回 0，不崩溃。"""
    store = _make_store()
    assert store.compact() == 0


# ── D. 工厂统一入口（company_id） ────────────────────────────────

def test_factory_get_store_with_company_id(monkeypatch, tmp_path):
    """get_store(company_id=...) 作为唯一入口工作（单例 + 类型）。"""
    monkeypatch.setattr("app.rag.vectorstore.factory.settings.rag_vector_store_path", str(tmp_path))
    clear_store_cache()
    try:
        s1 = get_store(company_id="xiaomi")
        s2 = get_store(company_id="xiaomi")
        assert s1 is s2
        assert isinstance(s1, FAISSStore)
    finally:
        clear_store_cache()


def test_factory_deprecated_get_vector_store_still_works(monkeypatch, tmp_path):
    """旧入口 get_vector_store() 继续可用（deprecated 兼容层）。"""
    from app.rag import vector_store as _vs
    from app.rag.vector_store import FAISSVectorStore, get_vector_store

    monkeypatch.setattr("app.rag.vector_store.settings.rag_vector_store_path", str(tmp_path))
    _vs._default_stores.pop("xiaomi", None)  # 隔离其他测试的旧缓存

    store = get_vector_store(company="xiaomi")

    assert isinstance(store, FAISSVectorStore)
    # 旧接口仍可检索（空库返回空，不崩溃）
    assert store.search([0.0] * DIM, company="xiaomi", top_k=3) == []


# ── E. 迁移协同：DenseRetriever / ingest 使用新 store ────────────

def test_dense_retriever_with_new_store():
    """DenseRetriever + 新 store → 返回 (DocumentChunk, score) 元组（管线兼容）。"""
    from app.rag.dense_retriever import DenseRetriever

    store = _make_store()
    store.add([_make_record("小米财务数据", "c1", company_id="xiaomi")])
    model = DummyEmbeddingModel(dim=DIM)

    results = DenseRetriever(model, store).search("财务", top_k=3, company="xiaomi")

    assert len(results) == 1
    chunk, score = results[0]
    assert isinstance(chunk, DocumentChunk)
    assert chunk.chunk_id == "c1"
    assert chunk.company == "xiaomi"
    assert 0.0 <= score <= 1.0


def test_ingestion_with_new_store(monkeypatch, tmp_path):
    """ingest() 经 get_store() 入库 → 新 store 可检索（迁移后生产路径）。"""
    from app.rag import ingest

    monkeypatch.setattr("app.rag.vectorstore.factory.settings.rag_vector_store_path", str(tmp_path))
    clear_store_cache()
    try:
        file_path = tmp_path / "小米年报.md"
        file_path.write_text(
            "# 小米集团 2024 年年报\n\n小米 2024 年营业收入 3659 亿元，净利润 236 亿元。",
            encoding="utf-8",
        )
        model = DummyEmbeddingModel(dim=DIM)

        chunks = ingest(str(file_path), company="xiaomi", _model=model)

        assert len(chunks) >= 1
        store = get_store(company_id="xiaomi")
        assert store.count() >= 1
        results = store.search(model.embed(["小米营收"])[0], top_k=5)
        assert len(results) >= 1
        assert all(r.metadata.get("company_id") == "xiaomi" for r in results)
    finally:
        clear_store_cache()
