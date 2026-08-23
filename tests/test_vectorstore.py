"""PR44.1 新向量库包测试（VectorStore ABC + FAISSStore wrapper + get_store 工厂）。

测试策略：
    - 使用真实 FAISSStore（包装旧 FAISSVectorStore，无 Mock）；
    - 向量由 DummyEmbeddingModel 生成（MD5 确定性，无语义——测试依赖
      过滤逻辑而非语义排序，故 top_k 取大保证全部召回再断言过滤结果）；
    - 每个测试创建独立实例，避免测试顺序依赖；
    - 工厂测试调用 clear_store_cache() 隔离单例缓存。
"""

import pytest

from app.rag.embedding import DummyEmbeddingModel
from app.rag.vectorstore import (
    FAISSStore,
    SearchResult,
    VectorRecord,
    VectorStore,
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


def _make_store(company: str = "xiaomi") -> FAISSStore:
    """创建独立的 FAISSStore 实例（非全局单例）。"""
    return FAISSStore(dim=DIM, company=company)


def _query(text: str) -> list[float]:
    return DummyEmbeddingModel(dim=DIM).embed([text])[0]


# ── A. 抽象接口 ───────────────────────────────────────────────────

def test_vector_store_abc_is_abstract():
    """新 VectorStore ABC 不可直接实例化。"""
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]


def test_vector_record_wraps_triple_consistency():
    """VectorRecord 封装 chunk_id + text + embedding + metadata 四要素。"""
    rec = _make_record("内容", "c1", year=2025)
    assert rec.chunk_id == "c1"
    assert rec.text == "内容"
    assert len(rec.embedding) == DIM
    assert rec.metadata["company_id"] == "xiaomi"
    assert rec.metadata["year"] == 2025


# ── B. add / search ───────────────────────────────────────────────

def test_add_and_search_returns_search_results():
    """入库 3 条 → search 返回 SearchResult 列表，全部召回。"""
    store = _make_store()
    records = [_make_record(f"小米年报内容 {i}", f"c{i}") for i in range(3)]
    store.add(records)

    results = store.search(_query("小米年报"), top_k=10)

    assert len(results) == 3
    for r in results:
        assert isinstance(r, SearchResult)
        assert r.chunk_id in {"c0", "c1", "c2"}
        assert 0.0 <= r.score <= 1.0
        assert r.metadata["company_id"] == "xiaomi"


def test_search_on_empty_store_returns_empty():
    """空库 search 返回空列表（不崩溃）。"""
    store = _make_store()
    assert store.search(_query("查询"), top_k=10) == []


def test_search_with_year_filter():
    """filters={"year": 2025} 只返回 year >= 2025 的记录。"""
    store = _make_store()
    store.add(
        [
            _make_record("2024 年报", "c2024", year=2024),
            _make_record("2025 年报", "c2025", year=2025),
            _make_record("2025 半年报", "c2025h", year=2025),
        ]
    )

    results = store.search(_query("年报"), top_k=10, filters={"year": 2025})

    assert {r.chunk_id for r in results} == {"c2025", "c2025h"}


def test_search_with_document_type_filter():
    """filters={"document_type": "annual_report"} 命中 source_type 别名。"""
    store = _make_store()
    store.add(
        [
            _make_record("年度财报", "c_ar", source_type="annual_report"),
            _make_record("券商研报", "c_rr", source_type="research_report"),
            _make_record("监管公告", "c_an", source_type="announcement"),
        ]
    )

    results = store.search(_query("财报"), top_k=10,
                           filters={"document_type": "annual_report"})

    assert {r.chunk_id for r in results} == {"c_ar"}


def test_search_with_combined_filters():
    """组合过滤：year + document_type 同时生效。"""
    store = _make_store()
    store.add(
        [
            _make_record("2024 年报", "a1", year=2024, source_type="annual_report"),
            _make_record("2025 年报", "a2", year=2025, source_type="annual_report"),
            _make_record("2025 研报", "b1", year=2025, source_type="research_report"),
        ]
    )

    results = store.search(
        _query("年报"), top_k=10,
        filters={"year": 2025, "document_type": "annual_report"},
    )

    assert {r.chunk_id for r in results} == {"a2"}


def test_search_with_company_filter_matching():
    """filters company_id 与 store 构造公司一致时正常返回。"""
    store = _make_store(company="xiaomi")
    store.add([_make_record("小米内容", "c0", company_id="xiaomi")])

    results = store.search(_query("小米"), top_k=10,
                           filters={"company_id": "xiaomi"})

    assert len(results) == 1
    assert results[0].chunk_id == "c0"


def test_search_with_wrong_company_returns_empty():
    """filters company_id 与 store 构造公司不同 → 空（公司隔离）。"""
    store = _make_store(company="xiaomi")
    store.add([_make_record("小米内容", "c0", company_id="xiaomi")])

    results = store.search(_query("小米"), top_k=10,
                           filters={"company_id": "catl"})

    assert results == []


# ── C. delete / update / count ────────────────────────────────────

def test_delete_logical_removes_from_search():
    """delete(ids) 后 search 不再返回已删 chunk。"""
    store = _make_store()
    store.add(
        [
            _make_record("保留内容", "keep"),
            _make_record("删除内容", "drop"),
        ]
    )

    removed = store.delete(["drop"])

    assert removed == 1
    results = store.search(_query("内容"), top_k=10)
    assert {r.chunk_id for r in results} == {"keep"}
    assert store.count() == 1


def test_delete_missing_id_returns_zero():
    """删除不存在的 chunk_id → 返回 0，不报错。"""
    store = _make_store()
    store.add([_make_record("内容", "c0")])
    assert store.delete(["不存在"]) == 0
    assert store.count() == 1


def test_update_replaces_record():
    """update(record) = 删旧 + 插新，返回旧记录是否存在。"""
    store = _make_store()
    store.add([_make_record("旧版本", "c1", year=2024)])

    updated = store.update(_make_record("新版本", "c1", year=2025))

    assert updated is True
    assert store.count() == 1  # 旧版本被逻辑删除，只留新版本活跃
    results = store.search(_query("版本"), top_k=10)
    assert len(results) == 1
    assert results[0].metadata["year"] == 2025
    assert results[0].text == "新版本"


def test_update_missing_record_returns_false():
    """update 不存在的 chunk_id → 返回 False（仍会插入）。"""
    store = _make_store()
    updated = store.update(_make_record("新内容", "new-id", year=2025))
    assert updated is False
    assert store.count() == 1


def test_count_reflects_active_records():
    """count() 随 add/delete 逐步变化。"""
    store = _make_store()
    assert store.count() == 0

    store.add([_make_record(f"内容 {i}", f"c{i}") for i in range(5)])
    assert store.count() == 5

    store.delete(["c0", "c1"])
    assert store.count() == 3


# ── D. validate_integrity ─────────────────────────────────────────

def test_validate_integrity_consistent_after_add():
    """add 后 faiss ntotal == metadata 条数 == active。"""
    store = _make_store()
    store.add([_make_record(f"内容 {i}", f"c{i}") for i in range(3)])

    integrity = store.validate_integrity()

    assert integrity["ntotal"] == 3
    assert integrity["metadata_count"] == 3
    assert integrity["active"] == 3
    assert integrity["deleted"] == 0
    assert integrity["consistent"] is True


def test_validate_integrity_reports_logical_delete():
    """delete 后 ntotal 不变，active 减少，deleted 增加。"""
    store = _make_store()
    store.add([_make_record(f"内容 {i}", f"c{i}") for i in range(3)])
    store.delete(["c1"])

    integrity = store.validate_integrity()

    assert integrity["ntotal"] == 3  # FAISS 索引未重建
    assert integrity["metadata_count"] == 3
    assert integrity["active"] == 2
    assert integrity["deleted"] == 1
    assert integrity["consistent"] is True


# ── E. 持久化 ─────────────────────────────────────────────────────

def test_save_load_roundtrip_preserves_data_and_deleted(tmp_path):
    """save → load 后恢复索引 + metadata + 逻辑删除标记。"""
    store = _make_store()
    store.add(
        [
            _make_record("保留内容", "keep", year=2025),
            _make_record("删除内容", "drop", year=2024),
        ]
    )
    store.delete(["drop"])
    store.save(str(tmp_path))

    loaded = FAISSStore(dim=DIM, company="xiaomi")
    loaded.load(str(tmp_path))

    integrity = loaded.validate_integrity()
    assert integrity["ntotal"] == 2
    assert integrity["active"] == 1
    assert integrity["deleted"] == 1
    assert integrity["consistent"] is True
    # 加载后 search 排除已删除
    results = loaded.search(_query("内容"), top_k=10)
    assert {r.chunk_id for r in results} == {"keep"}


# ── F. Hybrid / all_chunks / vector_of ────────────────────────────

def test_hybrid_search_delegates_to_dense():
    """hybrid_search(sparse=None) 结果与 search() 一致（Phase 1 退化）。"""
    store = _make_store()
    store.add(
        [
            _make_record("小米商业模式", "c0"),
            _make_record("小米财务数据", "c1"),
            _make_record("小米行业竞争", "c2"),
        ]
    )
    q = _query("商业模式")

    dense = store.search(q, top_k=3)
    hybrid = store.hybrid_search(q, None, top_k=3)

    assert [r.chunk_id for r in dense] == [r.chunk_id for r in hybrid]


def test_hybrid_search_with_sparse_raises():
    """hybrid_search(sparse 非 None) 抛 NotImplementedError（Phase 2 预留）。"""
    store = _make_store()
    store.add([_make_record("内容", "c0")])
    with pytest.raises(NotImplementedError, match="Phase 2"):
        store.hybrid_search(_query("内容"), [0.1] * DIM, top_k=3)


def test_all_chunks_excludes_deleted():
    """all_chunks() 排除已删除 chunk，且返回 VectorRecord。"""
    store = _make_store()
    store.add([_make_record(f"内容 {i}", f"c{i}") for i in range(3)])
    store.delete(["c1"])

    chunks = store.all_chunks()

    assert {c.chunk_id for c in chunks} == {"c0", "c2"}
    assert all(isinstance(c, VectorRecord) for c in chunks)


def test_vector_of_returns_embedding():
    """vector_of(chunk_id) 返回原始稠密向量。"""
    store = _make_store()
    store.add([_make_record("内容", "c0")])

    vec = store.vector_of("c0")

    assert vec is not None
    assert len(vec) == DIM
    assert store.vector_of("不存在") is None


# ── G. 工厂 ───────────────────────────────────────────────────────

def test_factory_get_store_singleton(monkeypatch, tmp_path):
    """get_store() 返回同一实例（按 backend:company 缓存）。"""
    monkeypatch.setattr("app.rag.vectorstore.factory.settings.rag_vector_store_path", str(tmp_path))
    clear_store_cache()
    try:
        s1 = get_store(company_id="xiaomi")
        s2 = get_store(company_id="xiaomi")
        assert s1 is s2
        assert isinstance(s1, FAISSStore)
    finally:
        clear_store_cache()


def test_factory_company_isolation(monkeypatch, tmp_path):
    """不同 company 返回独立实例。"""
    monkeypatch.setattr("app.rag.vectorstore.factory.settings.rag_vector_store_path", str(tmp_path))
    clear_store_cache()
    try:
        a = get_store(company_id="company_a")
        b = get_store(company_id="company_b")
        assert a is not b
        assert get_store(company_id="company_a") is a
    finally:
        clear_store_cache()


def test_factory_unknown_backend_raises():
    """未知 backend 抛 ValueError（milvus 已是合法后端，用任意非法值测试）。"""
    clear_store_cache()
    with pytest.raises(ValueError, match="未知后端"):
        get_store(company_id="xiaomi", backend="invalid_backend")


def test_factory_company_invalid_path_raises():
    """company 含非法路径字符拒绝（防路径穿越）。"""
    with pytest.raises(ValueError):
        get_store(company_id="../evil")
    with pytest.raises(ValueError):
        get_store(company_id="a/b")
