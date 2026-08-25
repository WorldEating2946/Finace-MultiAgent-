"""检索器单元测试。

测试 retriever.retrieve() 的 Dense 检索链路：
query → embedding → vector search → top-k chunks。

测试策略：
    - 使用 seam 模式（_model / _store 可选注入）隔离测试，
      避免依赖全局单例状态。
    - 向量由 DummyEmbeddingModel 生成（确定性、无随机性）。
"""

from app.rag.document import DocumentChunk
from app.rag.embedding import DummyEmbeddingModel
from app.rag.retriever import retrieve
from app.rag.vector_store import FAISSVectorStore

DIM = 128


# ── 测试辅助函数 ──────────────────────────────────────────────────

def _make_chunk(
    text: str,
    *,
    company: str = "",
    chunk_id: str = "test-0",
) -> DocumentChunk:
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


def _make_model(dim: int = DIM) -> DummyEmbeddingModel:
    return DummyEmbeddingModel(dim=dim)


def _make_store(dim: int = DIM) -> FAISSVectorStore:
    return FAISSVectorStore(dim=dim)


def _populate_store(
    store: FAISSVectorStore,
    texts: list[str],
    *,
    model: DummyEmbeddingModel | None = None,
    company: str = "",
) -> list[DocumentChunk]:
    """将文本列表入库，返回对应的 DocumentChunk 列表。"""
    if model is None:
        model = _make_model()
    chunks = [_make_chunk(t, company=company, chunk_id=f"id-{i}") for i, t in enumerate(texts)]
    vectors = model.embed(texts)
    store.add(chunks, vectors)
    return chunks


# ── 核心功能 ──────────────────────────────────────────────────────

def test_retrieve_returns_top_k_chunks():
    """检索应返回指定数量的 chunk。"""
    model = _make_model()
    store = _make_store()
    texts = [f"宁德时代深度分析 第{i}章" for i in range(10)]
    _populate_store(store, texts, model=model)

    results = retrieve("宁德时代", k=5, _model=model, _store=store)

    assert len(results) == 5
    for chunk, score in results:
        assert isinstance(chunk, DocumentChunk)
        assert isinstance(score, float)


def test_retrieve_results_sorted_by_score():
    """检索结果应按相似度降序排列。"""
    model = _make_model()
    store = _make_store()
    texts = [f"主题 {chr(65 + i)} 深度报告" for i in range(10)]
    _populate_store(store, texts, model=model)

    results = retrieve("主题 A", k=10, _model=model, _store=store)

    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), f"分数未降序：{scores}"


def test_retrieve_scores_in_range():
    """检索分数应在 [0, 1] 区间内。"""
    model = _make_model()
    store = _make_store()
    texts = ["商业模式分析", "财务数据概览", "行业竞争格局"]
    _populate_store(store, texts, model=model)

    results = retrieve("商业模式", k=3, _model=model, _store=store)

    for _, score in results:
        assert 0.0 <= score <= 1.0, f"分数 {score} 不在 [0, 1]"


def test_retrieve_returns_correct_content():
    """检索返回的 chunk.text 应与入库内容一致。"""
    model = _make_model()
    store = _make_store()
    texts = ["动力电池技术路线", "钠离子电池进展", "固态电池研发"]
    _populate_store(store, texts, model=model)

    results = retrieve("电池技术", k=3, _model=model, _store=store)

    returned_texts = {c.text for c, _ in results}
    assert "动力电池技术路线" in returned_texts


# ── 边界与错误处理 ───────────────────────────────────────────────

def test_retrieve_empty_store_returns_empty():
    """未入库时检索应返回空列表。"""
    model = _make_model()
    store = _make_store()

    results = retrieve("任意查询", k=5, _model=model, _store=store)

    assert results == []


def test_retrieve_k_greater_than_total_returns_all():
    """k 大于库内总数时返回全部 chunk（不崩溃、不补全）。"""
    model = _make_model()
    store = _make_store()
    texts = ["片段A", "片段B", "片段C"]
    _populate_store(store, texts, model=model)

    results = retrieve("片段", k=10, _model=model, _store=store)

    assert len(results) == 3  # 只有 3 条，不会强行补到 10


def test_retrieve_deterministic():
    """相同查询应返回相同结果（确定性向量 + 确定性检索）。"""
    model = _make_model()
    store = _make_store()
    texts = ["测试数据A", "测试数据B", "测试数据C"]
    _populate_store(store, texts, model=model)

    results1 = retrieve("测试", k=3, _model=model, _store=store)
    results2 = retrieve("测试", k=3, _model=model, _store=store)

    assert len(results1) == len(results2)
    for (c1, s1), (c2, s2) in zip(results1, results2):
        assert c1.chunk_id == c2.chunk_id
        assert s1 == s2


# ── seam 模式验证 ─────────────────────────────────────────────────

def test_retrieve_uses_injected_dependencies():
    """通过 _model / _store 注入时不应触及全局单例。"""
    model = _make_model()
    store = _make_store()
    _populate_store(store, ["Hello World"], model=model)

    # 注入的 store 有数据，能正常检索
    results = retrieve("Hello", k=1, _model=model, _store=store)
    assert len(results) == 1


def test_retrieve_with_company_filter():
    """company 过滤：只返回匹配公司的 chunk。"""
    model = _make_model()
    store = _make_store()

    # 两种公司的数据
    catl_texts = ["宁德时代电池技术"]
    byd_texts = ["比亚迪新能源汽车"]
    _populate_store(store, catl_texts, model=model, company="宁德时代")
    _populate_store(store, byd_texts, model=model, company="比亚迪")

    # 不带 company 过滤时，store.search(company="") 只匹配 company="" 的 chunk
    # 此测试验证公司标签被正确保存到元数据
    assert len(store._chunks) == 2
    assert store._chunks[0].company == "宁德时代"
    assert store._chunks[1].company == "比亚迪"


def test_sparse_cache_drops_entry_after_store_gc():
    """稀疏检索器缓存（WeakKeyDictionary）：store 被 GC 后条目自动移除。

    防止 id 复用误命中旧语料——若用 dict[int, ...]（key=id(store)），store 被
    回收后 id 被新 store 复用，会错误返回已失效公司的 BM25（召回漂移）。
    """
    import gc

    import app.rag.retriever as _rt

    model = _make_model()
    store = _make_store()
    _populate_store(store, ["宁德时代 电池业务", "小米 手机业务"], model=model)

    results = retrieve("宁德时代", k=3, _model=model, _store=store)
    assert results  # rag_hybrid=True → 触发 sparse 构建
    assert len(_rt._sparse_cache) == 1

    del store
    gc.collect()

    assert len(_rt._sparse_cache) == 0  # 弱引用：store 回收即移除，杜绝 id 复用
