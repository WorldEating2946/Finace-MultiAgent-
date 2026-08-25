"""PR44.3.1 Milvus Adapter 测试（第二个 VectorStore Contract 生产 Adapter）。

策略（对齐 test_faiss_adapter.py）：
    - 真实 MilvusStore，注入 ContractFakeMilvusClient（in-memory，无 pymilvus /
      无 Milvus 服务依赖——Windows 不支持 Milvus Lite，适配器测试必须零依赖）；
    - fake 用 numpy 余弦暴力扫描 = Milvus FLAT/COSINE 精确结果（AD-4）；
    - 向量由 DummyEmbeddingModel 生成（确定性），top_k 取大保证全量召回再断言过滤；
    - 每个测试独立实例，工厂测试用 clear_store_cache() 隔离单例；
    - 覆盖 VectorStore ABC（add/search/delete/update/count）+ Hybrid 支持
      （all_chunks/vector_of，AD-5 方法表——PR44.3.3 Benchmark 用真 retrieve() 管线需要）。

与 FAISSStore 的行为差异（本测试明确验证）：
    - add 幂等 = upsert 覆盖（FAISS 是跳过重复）；
    - delete = 物理删除（FAISS 是逻辑删除标记）；
    - 无 save()/load()（Milvus 持久化由服务端负责）。
"""

from app.rag.embedding import DummyEmbeddingModel
from app.rag.vectorstore import (
    MilvusStore,
    SearchResult,
    VectorRecord,
    clear_store_cache,
    get_store,
)
from tests.milvus_fake_client import ContractFakeMilvusClient

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


def _make_store(company: str = "xiaomi") -> MilvusStore:
    """创建独立的 MilvusStore（注入 fake client，非全局单例）。"""
    return MilvusStore(
        dim=DIM,
        uri="fake://unused",
        collection_name="test_finance_knowledge",
        company_id=company,
        client=ContractFakeMilvusClient(),
    )


def _query(text: str) -> list[float]:
    return DummyEmbeddingModel(dim=DIM).embed([text])[0]


# ── A. 契约检索（新接口 SearchResult + filters） ─────────────────

def test_add_and_search_returns_search_results():
    """add → search 返回 SearchResult 列表，数量一致。"""
    store = _make_store()
    store.add([_make_record(f"小米年报 {i}", f"c{i}") for i in range(3)])

    results = store.search(_query("小米年报"), top_k=10)

    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_search_on_empty_store_returns_empty():
    """空库 search/count 不崩溃，返回空/0（collection 尚未创建）。"""
    store = _make_store()
    assert store.search(_query("任何"), top_k=10) == []
    assert store.count() == 0


def test_search_filters_company_id():
    """filters={"company_id": "xiaomi"} 正确过滤（多公司隔离契约，AD-7）。"""
    store = _make_store(company="xiaomi")
    store.add([_make_record("小米内容", "c0", company_id="xiaomi")])

    hit = store.search(_query("小米"), top_k=10, filters={"company_id": "xiaomi"})
    miss = store.search(_query("小米"), top_k=10, filters={"company_id": "catl"})

    assert {r.chunk_id for r in hit} == {"c0"}
    assert miss == []


def test_search_filters_year():
    """filters={"year": 2025} → year >= 2025（数值过滤语义与 FAISS 一致）。"""
    store = _make_store()
    store.add([_make_record(f"年报 {i}", f"c{i}", year=2024 + i) for i in range(3)])

    results = store.search(_query("年报"), top_k=10, filters={"year": 2025})

    assert {r.chunk_id for r in results} == {"c1", "c2"}  # 2025、2026


def test_search_filters_document_type_alias():
    """document_type → metadata["source_type"] 别名映射（与 FAISS 语义一致）。"""
    store = _make_store()
    store.add([_make_record("研报内容", "c0", source_type="research_report")])
    store.add([_make_record("年报内容", "c1", source_type="annual_report")])

    results = store.search(
        _query("内容"), top_k=10, filters={"document_type": "annual_report"}
    )

    assert {r.chunk_id for r in results} == {"c1"}


def test_search_top_k():
    """top_k 限制返回条数。"""
    store = _make_store()
    store.add([_make_record(f"内容 {i}", f"c{i}") for i in range(5)])

    assert len(store.search(_query("内容"), top_k=2)) == 2


# ── B. 生命周期：delete / update / count ─────────────────────────

def test_add_idempotent_upsert_overwrites():
    """同 chunk_id 重复 add → upsert 覆盖（FAISS 是跳过重复，Milvus 是覆盖）。"""
    store = _make_store()
    store.add([_make_record("旧内容", "c1", year=2024)])
    store.add([_make_record("新内容", "c1", year=2025)])

    assert store.count() == 1
    results = store.search(_query("新内容"), top_k=10)
    assert len(results) == 1
    assert results[0].text == "新内容"
    assert results[0].metadata["year"] == 2025


def test_delete_then_search_absent():
    """insert → search 命中 → delete → search 不命中（物理删除）。"""
    store = _make_store()
    store.add([_make_record("保留", "keep"), _make_record("删除", "drop")])

    assert {r.chunk_id for r in store.search(_query("内容"), top_k=10)} == {"keep", "drop"}

    removed = store.delete(["drop"])

    assert removed == 1
    assert store.count() == 1
    assert {r.chunk_id for r in store.search(_query("内容"), top_k=10)} == {"keep"}


def test_delete_returns_count_for_missing():
    """delete 含不存在的 id → 只统计实际删除数。"""
    store = _make_store()
    store.add([_make_record("内容", "c0")])

    assert store.delete(["c0", "c999"]) == 1
    assert store.delete(["c999"]) == 0


def test_delete_company_isolation():
    """delete 带公司过滤：不删共享 collection 中其他公司的 chunk（AD-7 防御）。

    chunk_id 是共享 collection 的主键（AD-3），两公司不可能同 id 同库——所以
    通过直接向 collection 注入他司数据（绕过 store 的 add）来验证 company 过滤。
    """
    client = ContractFakeMilvusClient()
    store = MilvusStore(
        dim=DIM, uri="fake://unused", collection_name="test_finance_knowledge",
        company_id="xiaomi", client=client,
    )
    store.add([_make_record("小米内容", "c0", company_id="xiaomi")])
    # 模拟共享 collection 中其他公司的数据
    client.upsert(
        "test_finance_knowledge",
        [{"chunk_id": "catl-0", "company_id": "catl", "text": "宁德内容",
          "embedding": _query("宁德内容"), "metadata": {"company_id": "catl"}}],
    )

    removed = store.delete(["catl-0"])

    assert removed == 0  # xiaomi store 不删 catl 数据
    # 全库计数（client 层）：两条都在（count() 是公司作用域=1，这里用 client 验证隔离）
    all_rows = client.query("test_finance_knowledge", output_fields=["chunk_id"])
    assert len(all_rows) == 2
    assert store.count() == 1  # 公司作用域：只数本公司


def test_update_then_search_returns_new():
    """insert old → update → search 返回 new chunk（旧内容不再出现）。"""
    store = _make_store()
    store.add([_make_record("旧版本内容", "c1", year=2024)])

    updated = store.update(_make_record("新版本内容", "c1", year=2025))

    assert updated is True
    assert store.count() == 1
    results = store.search(_query("版本"), top_k=10)
    assert len(results) == 1
    assert results[0].text == "新版本内容"
    assert results[0].metadata["year"] == 2025


def test_update_missing_record_inserts():
    """update 不存在的记录 → 返回 False 并插入。"""
    store = _make_store()

    updated = store.update(_make_record("新内容", "c_new"))

    assert updated is False
    assert store.count() == 1


def test_count_reflects_add_delete():
    """count 跟随 add/delete 变化（活跃记录数）。"""
    store = _make_store()
    store.add([_make_record(f"内容 {i}", f"c{i}") for i in range(3)])
    assert store.count() == 3

    store.delete(["c0"])
    assert store.count() == 2

    store.delete(["c1", "c2"])
    assert store.count() == 0


# ── C. 与 FAISS 的接口差异 ───────────────────────────────────────

def test_store_has_no_local_persistence():
    """Milvus 持久化由服务端负责——不提供 save/load（区别于 FAISSStore）。"""
    store = _make_store()
    assert not hasattr(store, "save")
    assert not hasattr(store, "load")


# ── E. Hybrid 支持（all_chunks / vector_of，AD-5 方法表）─────────

def test_all_chunks_returns_company_chunks():
    """all_chunks 只返回本公司 chunk（共享 collection 多公司过滤）。"""
    store = _make_store(company="xiaomi")
    store.add(
        [
            _make_record("小米a", "xa", company_id="xiaomi"),
            _make_record("小米b", "xb", company_id="xiaomi"),
            _make_record("宁德a", "ca", company_id="catl"),
        ]
    )
    chunks = store.all_chunks()
    assert [c.chunk_id for c in chunks] == ["xa", "xb"]
    assert all(c.metadata["company_id"] == "xiaomi" for c in chunks)


def test_all_chunks_empty_when_no_collection():
    """collection 不存在 → 空列表（不崩）。"""
    store = _make_store()
    assert store.all_chunks() == []


def test_vector_of_returns_embedding():
    """vector_of 返回指定 chunk 的稠密向量。"""
    store = _make_store()
    rec = _make_record("内容", "c0")
    store.add([rec])
    assert store.vector_of("c0") == rec.embedding


def test_vector_of_missing_returns_none():
    """vector_of 查不存在的 chunk → None（不崩）。"""
    store = _make_store()
    assert store.vector_of("nope") is None


# ── D. 工厂统一入口（backend="milvus"） ─────────────────────────

def test_factory_get_store_milvus():
    """get_store(backend="milvus") 返回 MilvusStore 单例（懒加载，无需 pymilvus）。"""
    clear_store_cache()
    try:
        s1 = get_store(company_id="xiaomi", backend="milvus")
        s2 = get_store(company_id="xiaomi", backend="milvus")
        assert s1 is s2
        assert isinstance(s1, MilvusStore)
        # 不同公司隔离单例
        s3 = get_store(company_id="catl", backend="milvus")
        assert s3 is not s1
    finally:
        clear_store_cache()
