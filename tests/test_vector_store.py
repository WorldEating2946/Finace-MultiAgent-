"""向量库抽象层单元测试。

测试 VectorStore ABC + FAISSVectorStore 实现 + get_vector_store 单例。

测试策略：
    - 所有测试使用真实 FAISSVectorStore（无 Mock），
      向量由 DummyEmbeddingModel 生成（确定性、无随机性）。
    - 每个测试创建独立实例，避免测试顺序依赖。
"""

import pytest

from app.rag.document import DocumentChunk
from app.rag.embedding import DummyEmbeddingModel
from app.rag.vector_store import (
    FAISSVectorStore,
    VectorStore,
    get_vector_store,
)

DIM = 128


# ── 测试辅助函数 ──────────────────────────────────────────────────

def _make_chunk(
    text: str,
    *,
    company: str = "宁德时代",
    chunk_id: str = "test-0",
) -> DocumentChunk:
    """创建测试用 DocumentChunk（常规默认值）。"""
    return DocumentChunk(
        chunk_id=chunk_id,
        company=company,
        doc_type="text",
        source="/tmp/test.txt",
        source_name="测试文档",
        page=0,
        text=text,
        metadata={},
    )


def _make_vectors(texts: list[str], dim: int = DIM) -> list[list[float]]:
    """使用 DummyEmbeddingModel 生成确定性向量。"""
    return DummyEmbeddingModel(dim=dim).embed(texts)


def _make_store(dim: int = DIM) -> FAISSVectorStore:
    """创建独立的 FAISSVectorStore 实例（非全局单例）。"""
    return FAISSVectorStore(dim=dim)


# ── A. 抽象接口 ───────────────────────────────────────────────────

def test_vector_store_is_abstract():
    """VectorStore 为抽象类，不可直接实例化。"""
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]


# ── B. 核心操作 ───────────────────────────────────────────────────

def test_add_and_search_returns_matching_chunks():
    """入库 5 个 chunk 后检索，应返回匹配结果。"""
    store = _make_store()
    texts = [f"宁德时代测试文本片段 {i}" for i in range(5)]
    chunks = [_make_chunk(t, chunk_id=f"id-{i}") for i, t in enumerate(texts)]
    vectors = _make_vectors(texts)

    store.add(chunks, vectors)

    query_vec = _make_vectors(["宁德时代"])[0]
    results = store.search(query_vec, company="宁德时代", top_k=3)

    assert len(results) >= 1
    assert len(results) <= 3
    for chunk, score in results:
        assert isinstance(chunk, DocumentChunk)
        assert isinstance(score, float)
        assert "宁德时代" in chunk.text


def test_search_filters_by_company():
    """检索时按 company 过滤，不返回其他公司的 chunk。"""
    store = _make_store()

    # 宁德时代 3 条
    catl_texts = [f"宁德时代资料 {i}" for i in range(3)]
    catl_chunks = [_make_chunk(t, company="宁德时代", chunk_id=f"catl-{i}") for i, t in enumerate(catl_texts)]

    # 比亚迪 3 条
    byd_texts = [f"比亚迪资料 {i}" for i in range(3)]
    byd_chunks = [_make_chunk(t, company="比亚迪", chunk_id=f"byd-{i}") for i, t in enumerate(byd_texts)]

    all_chunks = catl_chunks + byd_chunks
    all_texts = catl_texts + byd_texts
    store.add(all_chunks, _make_vectors(all_texts))

    query_vec = _make_vectors(["动力电池"])[0]
    results = store.search(query_vec, company="宁德时代", top_k=5)

    assert len(results) >= 1
    for chunk, _ in results:
        assert chunk.company == "宁德时代", f"不应返回 {chunk.company} 的 chunk"


def test_search_on_empty_store_returns_empty_list():
    """未入库时检索，返回空列表（不崩溃）。"""
    store = _make_store()
    query_vec = _make_vectors(["测试查询"])[0]
    results = store.search(query_vec, company="宁德时代")
    assert results == []


def test_add_empty_chunks_is_safe_noop():
    """传入空列表不抛异常，且 store 保持为空。"""
    store = _make_store()
    store.add([], [])
    query_vec = _make_vectors(["测试"])[0]
    assert store.search(query_vec, company="宁德时代") == []


def test_search_returns_correct_text_content():
    """检索返回的 chunk.text 与入库时一致（索引映射正确）。"""
    store = _make_store()
    texts = ["商业模式分析", "财务数据概览", "行业竞争格局"]
    chunks = [_make_chunk(t, chunk_id=f"id-{i}") for i, t in enumerate(texts)]
    store.add(chunks, _make_vectors(texts))

    query_vec = _make_vectors(["商业模式"])[0]
    results = store.search(query_vec, company="宁德时代", top_k=3)

    returned_texts = {c.text for c, _ in results}
    assert "商业模式分析" in returned_texts


# ── C. 错误处理与边界 ─────────────────────────────────────────────

def test_add_mismatched_lengths_raises_value_error():
    """chunks 与 vectors 长度不一致时抛出 ValueError。"""
    store = _make_store()
    chunks = [_make_chunk("文本 A"), _make_chunk("文本 B")]
    vectors = _make_vectors(["仅一条"])

    with pytest.raises(ValueError, match="长度不一致"):
        store.add(chunks, vectors)


def test_hybrid_search_with_sparse_raises_not_implemented():
    """sparse_vector 非 None 时抛出 NotImplementedError（Phase 2 预留）。"""
    store = _make_store()
    chunk = _make_chunk("测试内容")
    store.add([chunk], _make_vectors(["测试内容"]))

    dense_vec = _make_vectors(["测试"])[0]
    sparse_vec = [0.1] * DIM  # 模拟 sparse 向量

    with pytest.raises(NotImplementedError, match="Phase 2"):
        store.hybrid_search(dense_vec, sparse_vec, company="宁德时代")


def test_search_with_nonexistent_company_returns_empty():
    """检索不存在的 company 时返回空列表。"""
    store = _make_store()
    chunks = [_make_chunk("宁德时代资料", company="宁德时代", chunk_id="catl-0")]
    store.add(chunks, _make_vectors(["宁德时代资料"]))

    query_vec = _make_vectors(["资料"])[0]
    results = store.search(query_vec, company="不存在的公司")
    assert results == []


# ── D. 数据完整性 ─────────────────────────────────────────────────

def test_search_scores_are_in_zero_to_one_range():
    """检索分数应映射到 [0, 1] 区间。"""
    store = _make_store()
    texts = [f"片段 {i}" for i in range(5)]
    chunks = [_make_chunk(t, chunk_id=f"id-{i}") for i, t in enumerate(texts)]
    store.add(chunks, _make_vectors(texts))

    query_vec = _make_vectors(["片段"])[0]
    results = store.search(query_vec, company="宁德时代", top_k=5)

    for _, score in results:
        assert 0.0 <= score <= 1.0, f"分数 {score} 不在 [0, 1] 区间"


def test_search_results_sorted_by_score_descending():
    """检索结果应按分数降序排列。"""
    store = _make_store()
    texts = [f"主题 {chr(65 + i)} 的详细内容" for i in range(10)]
    chunks = [_make_chunk(t, chunk_id=f"id-{i}") for i, t in enumerate(texts)]
    store.add(chunks, _make_vectors(texts))

    query_vec = _make_vectors(["主题 A"])[0]
    results = store.search(query_vec, company="宁德时代", top_k=10)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), f"分数未降序：{scores}"


def test_hybrid_search_with_sparse_none_delegates_to_dense():
    """hybrid_search(sparse_vector=None) 结果应与 search() 一致。"""
    store = _make_store()
    texts = ["宁德时代商业模式", "动力电池技术路线", "新能源政策解读"]
    chunks = [_make_chunk(t, chunk_id=f"id-{i}") for i, t in enumerate(texts)]
    store.add(chunks, _make_vectors(texts))

    query_vec = _make_vectors(["商业模式"])[0]

    dense_results = store.search(query_vec, company="宁德时代", top_k=3)
    hybrid_results = store.hybrid_search(query_vec, None, company="宁德时代", top_k=3)

    assert len(dense_results) == len(hybrid_results)
    for (dc, ds), (hc, hs) in zip(dense_results, hybrid_results):
        assert dc.chunk_id == hc.chunk_id
        assert ds == hs


# ── 单例入口 ──────────────────────────────────────────────────────

def test_get_vector_store_returns_singleton():
    """get_vector_store() 应返回同一实例，且类型为 FAISSVectorStore。"""
    # 注意：此测试可能受其他测试中创建的 FAISSVectorStore 影响，
    # 仅验证单例行为与返回类型。
    store1 = get_vector_store()
    store2 = get_vector_store()
    assert store1 is store2
    assert isinstance(store1, FAISSVectorStore)


# ── E. 持久化（save / load）─────────────────────────────────────

def test_save_creates_index_and_metadata_files(tmp_path):
    """save() 应生成 index.faiss 与 metadata.json 两个文件。"""
    store = _make_store()
    texts = ["宁德时代测试文本"]
    store.add([_make_chunk(t) for t in texts], _make_vectors(texts))

    store.save(str(tmp_path))

    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "metadata.json").exists()


def test_save_load_roundtrip_restores_data(tmp_path):
    """save → load 后应完整恢复索引、chunk 元数据与检索能力。"""
    store = _make_store()
    texts = ["宁德时代动力电池龙头", "比亚迪新能源汽车", "基金赎回T+1"]
    chunks = [
        _make_chunk(t, company="宁德时代", chunk_id=f"id-{i}")
        for i, t in enumerate(texts)
    ]
    chunks[0].metadata = {"行业": "新能源"}
    store.add(chunks, _make_vectors(texts))
    store.save(str(tmp_path))

    loaded = FAISSVectorStore()
    loaded.load(str(tmp_path))

    assert loaded._index is not None
    assert loaded._index.ntotal == 3
    assert len(loaded._chunks) == 3
    assert loaded._chunks[0].text == texts[0]
    assert loaded._chunks[0].company == "宁德时代"
    assert loaded._chunks[0].metadata == {"行业": "新能源"}

    # 加载后可正常检索（DummyEmbedding 伪向量不保证语义排序，只验证检索能力恢复）
    query_vec = _make_vectors(["动力电池"])[0]
    results = loaded.search(query_vec, company="宁德时代", top_k=2)
    assert len(results) >= 1
    assert all(c.text in texts for c, _ in results)


def test_load_missing_files_raises(tmp_path):
    """目录存在但存档文件缺失时应抛 FileNotFoundError。"""
    store = FAISSVectorStore()
    with pytest.raises(FileNotFoundError):
        store.load(str(tmp_path))


def test_save_without_dir_is_safe_noop():
    """未配置目录时 save() 应静默跳过，不报错。"""
    store = FAISSVectorStore()
    store.add([_make_chunk("内容")], _make_vectors(["内容"]))
    store.save()  # 不应抛异常


def test_load_without_dir_is_safe_noop():
    """未配置目录时 load() 应静默跳过，不报错。"""
    FAISSVectorStore().load()


def test_get_vector_store_autoloads_saved_index(monkeypatch, tmp_path):
    """get_vector_store() 启动时应自动加载已保存的存档。"""
    import app.rag.vector_store as vs

    # 准备一份持久化存档（写入 tmp_path 根目录，对应 company=""）
    store = FAISSVectorStore(dim=DIM, dir_path=tmp_path)
    texts = ["宁德时代动力电池"]
    store.add([_make_chunk(t) for t in texts], _make_vectors(texts))
    store.save()

    monkeypatch.setattr(vs.settings, "rag_vector_store_path", str(tmp_path))
    monkeypatch.setattr(vs, "_default_stores", {})

    loaded = get_vector_store()
    assert isinstance(loaded, FAISSVectorStore)
    assert loaded._index is not None and loaded._index.ntotal == 1
    assert loaded._chunks[0].text == "宁德时代动力电池"


# ── F. 多公司知识库隔离 ────────────────────────────────────────

def test_get_vector_store_isolates_by_company(monkeypatch, tmp_path):
    """不同 company 应返回独立 store，并绑定各自的持久化目录。"""
    import app.rag.vector_store as vs

    monkeypatch.setattr(vs.settings, "rag_vector_store_path", str(tmp_path))
    monkeypatch.setattr(vs, "_default_stores", {})

    store_a = get_vector_store(company="company_a")
    store_b = get_vector_store(company="company_b")

    assert store_a is not store_b
    assert store_a._dir_path == tmp_path / "company_a"
    assert store_b._dir_path == tmp_path / "company_b"

    # 同一 company 复用同一实例
    assert get_vector_store(company="company_a") is store_a


def test_get_vector_store_company_invalid_path_raises(monkeypatch):
    """company 含路径分隔符或 '..' 时应拒绝，防止越级访问。"""
    import app.rag.vector_store as vs

    monkeypatch.setattr(vs, "_default_stores", {})

    with pytest.raises(ValueError):
        get_vector_store(company="../evil")
    with pytest.raises(ValueError):
        get_vector_store(company="a/b")


def test_save_load_with_chinese_company_dir(monkeypatch, tmp_path):
    """中文 company 目录名应可正常持久化（Windows 非 ASCII 路径回归）。

    faiss.write_index 在 Windows 上对中文路径转码出错，
    已改为 serialize + Python 写盘；本测试防止该问题回归。
    """
    import app.rag.vector_store as vs

    monkeypatch.setattr(vs.settings, "rag_vector_store_path", str(tmp_path))
    monkeypatch.setattr(vs, "_default_stores", {})

    store = get_vector_store(company="测试公司")
    texts = ["宁德时代动力电池"]
    store.add([_make_chunk(t, company="测试公司") for t in texts], _make_vectors(texts))
    store.save()

    loaded = FAISSVectorStore(dir_path=tmp_path / "测试公司")
    loaded.load()
    assert loaded._index is not None and loaded._index.ntotal == 1
    assert loaded._chunks[0].company == "测试公司"
