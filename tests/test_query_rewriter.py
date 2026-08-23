"""Query Rewrite 测试：规则扩展 + LLM 改写 + multi-query 检索集成。

单元测试覆盖 rewrite() 的扩展 / 直通 / 去重 / LLM 回退；
集成测试验证词汇鸿沟（用户口语 vs 年报正式用语）经重写后可被召回。
"""

from app.rag.document import DocumentChunk
from app.rag.embedding import DummyEmbeddingModel
from app.rag.query import LLMQueryRewriter, get_query_rewriter
from app.rag.query.rewriter import QueryRewriter, RuleBasedQueryRewriter
from app.rag.reranker.dummy import DummyReranker
from app.rag.retriever import retrieve
from app.rag.vector_store import FAISSVectorStore


def _chunk(cid: str, text: str, company: str = "测试公司") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid, company=company, doc_type="text",
        source="/tmp/t.txt", source_name="t", page=0, text=text,
    )


# ── 单元测试：rewrite() ─────────────────────────────────────────

def test_rewrite_expands_car_strategy():
    """'小米汽车业务未来战略' → 含年报正式用语'智能电动汽车/新能源汽车'变体。"""
    r = RuleBasedQueryRewriter()
    result = r.rewrite("小米汽车业务未来战略")

    assert result[0] == "小米汽车业务未来战略"  # 原始保留
    assert any("智能电动汽车" in q for q in result)
    assert any("新能源汽车" in q for q in result)


def test_rewrite_expands_rd():
    """'公司研发投入金额' → 含'研发费用'变体（年报财务术语）。"""
    result = RuleBasedQueryRewriter().rewrite("公司研发投入金额")
    assert any("研发费用" in q for q in result)


def test_rewrite_expands_revenue():
    """'小米2025年营业收入' → 含'主营业务收入'变体。"""
    result = RuleBasedQueryRewriter().rewrite("小米2025年营业收入")
    assert any("主营业务收入" in q for q in result)


def test_rewrite_expands_smartphone_shipment():
    """'小米智能手机全球出货量' → 含'全球销量/智能终端'变体。"""
    result = RuleBasedQueryRewriter().rewrite("小米智能手机全球出货量")
    assert any("全球销量" in q for q in result)
    assert any("智能终端" in q for q in result)


def test_rewrite_passthrough_precise_query():
    """CATL 精确查询（无同义词命中）→ 直通 [query]，不产生额外查询。"""
    q = "宁德时代2025年归属于上市公司股东的净利润"
    result = RuleBasedQueryRewriter().rewrite(q)
    assert result == [q]


def test_rewrite_avoids_synonym_doubling():
    """'现金及现金等价物' 已含同义词，不得扩展出重复叠加变体。"""
    result = RuleBasedQueryRewriter().rewrite("现金及现金等价物")
    assert result[0] == "现金及现金等价物"
    assert all("现金及现金等价物及" not in q for q in result)  # 无叠加


def test_rewrite_bounded_expansions():
    """变体总数有上限（默认 4，含原始），避免召回成本失控。"""
    r = RuleBasedQueryRewriter()
    # 同时命中 汽车/智能手机/全球出货量/出货量 的复合查询
    result = r.rewrite("小米汽车业务和智能手机全球出货量")
    assert len(result) <= 4


class _PassThroughRewriter(QueryRewriter):
    """注入 seam：直通，不扩展（验证 rewrite 被正确调用/可注入）。"""

    def rewrite(self, query: str) -> list[str]:
        return [query]


# ── LLM 改写器 ─────────────────────────────────────────────────

def _llm_rewriter(call_fn, api_key="test-key"):
    """构造带 mock LLM 调用的 LLMQueryRewriter（隔离网络）。"""
    return LLMQueryRewriter(
        api_key=api_key, base_url="https://mock.local", model="mock",
        _call=call_fn,
    )


def test_llm_rewrite_uses_variants():
    """LLM 返回变体 → 结果 = [原始] + 变体。"""
    r = _llm_rewriter(lambda prompt: '["智能电动汽车业务发展", "新能源汽车规划"]')
    result = r.rewrite("小米汽车业务未来战略")

    assert result[0] == "小米汽车业务未来战略"  # 原始保留
    assert "智能电动汽车业务发展" in result
    assert "新能源汽车规划" in result


def test_llm_rewrite_parses_json_codeblock():
    """LLM 输出带 ```json``` 包裹 → 仍能解析。"""
    content = '```json\n["研发费用情况", "研发支出明细"]\n```'
    r = _llm_rewriter(lambda prompt: content)
    result = r.rewrite("公司研发投入金额")

    assert "研发费用情况" in result


def test_llm_rewrite_original_first_and_dedup():
    """原始 query 在前；与原始相同的变体去重。"""
    r = _llm_rewriter(lambda prompt: '["公司研发投入金额", "研发费用"]')
    result = r.rewrite("公司研发投入金额")

    assert result[0] == "公司研发投入金额"
    assert result.count("公司研发投入金额") == 1
    assert "研发费用" in result


def test_llm_rewrite_fallback_on_call_error():
    """LLM 调用抛异常 → 回退规则版（不崩溃，仍返回可用变体）。"""
    def boom(prompt):
        raise RuntimeError("network down")

    r = _llm_rewriter(boom)
    result = r.rewrite("公司研发投入金额")

    assert result[0] == "公司研发投入金额"
    assert len(result) >= 1  # 回退到规则扩展（含"研发费用"变体）
    assert any("研发费用" in q for q in result)


def test_llm_rewrite_fallback_without_api_key():
    """无 API key → 直接回退规则版，不发网络请求。"""
    called = []
    r = LLMQueryRewriter(
        api_key="", base_url="https://mock", model="mock",
        _call=lambda prompt: called.append(prompt) or "[]",
    )
    result = r.rewrite("小米汽车业务未来战略")

    assert called == []  # 未发请求
    assert "智能电动汽车" in result[1]  # 规则版扩展


def test_llm_rewrite_fallback_on_bad_output():
    """LLM 输出无法解析 → 回退规则版。"""
    r = _llm_rewriter(lambda prompt: "抱歉，我无法回答。")
    result = r.rewrite("小米汽车业务未来战略")

    assert "智能电动汽车" in result[1]  # 回退规则版


def test_get_query_rewriter_factory_modes(monkeypatch):
    """工厂按 settings.rag_query_rewriter 返回对应改写器。"""
    from app.core.config import settings

    monkeypatch.setattr(settings, "rag_query_rewriter", "rule")
    assert isinstance(get_query_rewriter(), RuleBasedQueryRewriter)

    monkeypatch.setattr(settings, "rag_query_rewriter", "off")
    off = get_query_rewriter()
    assert off.rewrite("小米汽车业务") == ["小米汽车业务"]  # 直通

    monkeypatch.setattr(settings, "rag_query_rewriter", "llm")
    assert isinstance(get_query_rewriter(), LLMQueryRewriter)


# ── 集成测试：multi-query 检索 ───────────────────────────────────

def test_multi_query_bridges_vocabulary_gap():
    """词汇鸿沟：原始 query 单独难命中，重写后经扩展 query（BM25）召回。

    DummyEmbeddingModel 为 MD5 确定性向量，dense 不体现语义相似；
    rewrite 扩展出与 chunk 文本共享词（智能电动汽车）的变体，
    靠 BM25 精确召回 → 验证 multi-query 融合通路有效。
    """
    model = DummyEmbeddingModel(dim=8)
    store = FAISSVectorStore(dim=8)
    texts = [
        "报告期内公司智能电动汽车业务营收增长迅速",
        "公司智能手机全球销量再创新高",
        "公司研发费用达到101亿元",
    ]
    store.add([_chunk(f"id-{i}", t) for i, t in enumerate(texts)], model.embed(texts))

    results = retrieve(
        "小米汽车业务", k=3, company="测试公司", _model=model, _store=store
    )

    assert len(results) >= 1
    found = [c.text for c, _ in results]
    assert any("智能电动汽车" in t for t in found), f"未召回: {found}"


def test_multi_query_uses_original_for_rerank():
    """reranker 收到的应是原始 query（扩展只负责召回，精排以用户意图为准）。"""

    class _RecordingReranker(DummyReranker):
        def __init__(self):
            self.queries: list[str] = []

        def rerank(self, query, chunks):
            self.queries.append(query)
            return list(chunks)

    model = DummyEmbeddingModel(dim=8)
    store = FAISSVectorStore(dim=8)
    texts = ["公司智能电动汽车业务营收增长迅速", "公司智能手机全球销量再创新高"]
    store.add([_chunk(f"id-{i}", t) for i, t in enumerate(texts)], model.embed(texts))

    reranker = _RecordingReranker()
    original = "小米汽车业务"
    retrieve(
        original, k=2, company="测试公司",
        _model=model, _store=store, _reranker=reranker,
    )

    assert len(reranker.queries) == 1  # 精排只跑一次
    assert reranker.queries[0] == original  # 用原始 query，而非扩展变体


def test_multi_query_rewriter_seam_injectable():
    """_rewriter 注入直通版 → 走单查询路径，结果与不注入等价可测。"""
    model = DummyEmbeddingModel(dim=8)
    store = FAISSVectorStore(dim=8)
    texts = ["宁德时代动力电池系统销量全球第一", "公司研发投入持续增长"]
    store.add([_chunk(f"id-{i}", t) for i, t in enumerate(texts)], model.embed(texts))

    results = retrieve(
        "宁德时代", k=2, company="测试公司",
        _model=model, _store=store, _rewriter=_PassThroughRewriter(),
    )

    assert len(results) >= 1
    for _, score in results:
        assert 0.0 <= score <= 1.0
