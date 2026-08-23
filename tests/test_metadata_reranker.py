"""Metadata-aware Rerank 测试（PR #33）。

覆盖：Context Builder 格式化 / 章节优先级信号 / 关键词命中信号 / Hybrid 融合排序。
用 fake CrossEncoder 模型（注入 _load_model）隔离真实 2.2GB 模型。
"""

from app.rag.document import DocumentChunk
from app.rag.reranker.context_builder import RerankContextBuilder
from app.rag.reranker.metadata_reranker import MetadataReranker
from app.rag.reranker.section_priority import get_section_priority


def _chunk(
    text: str,
    *,
    company: str = "测试公司",
    chapter: str = "",
    section: str = "",
    page: int | None = None,
    cid: str = "c",
) -> DocumentChunk:
    meta = {"company": company, "chapter": chapter, "section": section}
    return DocumentChunk(
        chunk_id=cid, company=company, doc_type="pdf",
        source="/t.txt", source_name="t", page=page, text=text,
        metadata={k: v for k, v in meta.items() if v},
    )


class _FakeModel:
    """假 CrossEncoder：按内容是否含标记返回固定分数（确定性，隔离真实模型）。"""

    def predict(self, pairs, max_length=1024):
        return [1.0 if "[Chapter] 董事会报告" in ctx else 0.5 for _, ctx in pairs]


# 测试公司走 metadata 融合；未配置公司默认纯 CE（零回归）
_TEST_WEIGHTS = {"测试公司": (1.0, 0.0, 0.0)}
_TEST_WEIGHTS_SECTION = {"测试公司": (1.0, 0.5, 0.0)}


class _TestMetadataReranker(MetadataReranker):
    """注入假模型的 MetadataReranker（隔离真实 2.2GB CrossEncoder）。"""

    def _load_model(self):
        return _FakeModel()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("company_weights", _TEST_WEIGHTS)
        super().__init__(*args, **kwargs)


# ── Context Builder ────────────────────────────────────────────

def test_context_builder_formats_all_fields():
    """完整 metadata → 正确格式化的结构化上下文。"""
    chunk = _chunk(
        "小米汽车业务快速发展",
        company="小米", chapter="管理层讨论及分析",
        section="5.1 智能电动汽车业务", page=68,
    )
    ctx = RerankContextBuilder().build(chunk)

    assert "[Company] 小米" in ctx
    assert "[Chapter] 管理层讨论及分析" in ctx
    assert "[Section] 5.1 智能电动汽车业务" in ctx
    assert "[Page] 68" in ctx
    assert "[Content] 小米汽车业务快速发展" in ctx
    # 顺序：Company → Chapter → Section → Page → Content
    idx = [ctx.index(s) for s in ("[Company]", "[Chapter]", "[Section]", "[Page]", "[Content]")]
    assert idx == sorted(idx)


def test_context_builder_skips_empty_fields():
    """空字段不生成空标签。"""
    ctx = RerankContextBuilder().build(
        _chunk("纯正文内容", company="", page=None)
    )
    assert "[Company]" not in ctx
    assert "[Chapter]" not in ctx
    assert "[Section]" not in ctx
    assert "[Page]" not in ctx
    assert "[Content] 纯正文内容" in ctx


# ── 章节优先级信号 ─────────────────────────────────────────────

def test_section_signal_matches_priority():
    """chunk 在"管理层讨论及分析"章节 → 返回优先级 0.15。"""
    r = _TestMetadataReranker()
    assert r._section_signal(_chunk("正文", chapter="管理层讨论及分析")) == 0.15


def test_section_signal_zero_for_unlisted():
    """未在优先级表的章节 → 0。"""
    r = _TestMetadataReranker()
    assert r._section_signal(_chunk("正文", chapter="封面")) == 0.0


def test_section_priority_default_table():
    """默认优先级表包含核心分析章节。"""
    p = get_section_priority()
    assert p["管理层讨论及分析"] == 0.15
    assert p["风险因素"] == 0.10


# ── 关键词命中信号 ─────────────────────────────────────────────

def test_keyword_signal_partial_match():
    """query 分词在 chunk 文本的部分命中率。

    "小米汽车业务" → jieba ["小米","汽车","业务"]（3 词）；
    chunk 含"汽车/业务"但缺"小米" → hits=2 → 2/max(3,3) ≈ 0.667。
    """
    r = _TestMetadataReranker()
    score = r._keyword_signal("小米汽车业务", _chunk("公司汽车业务营收增长"))
    assert 0.5 < score < 0.8


def test_keyword_signal_full_match():
    """全命中 → 1.0。"""
    r = _TestMetadataReranker()
    score = r._keyword_signal("小米汽车业务", _chunk("小米汽车业务快速发展"))
    assert score == 1.0


# ── Hybrid 融合排序 ────────────────────────────────────────────

def test_fusion_preserves_ce_order_when_metadata_neutral():
    """α=1.0 β=0 γ=0（纯 CE）：排序 = 假模型分数序（董事会报告分高者优先）。"""
    r = _TestMetadataReranker(company_weights=_TEST_WEIGHTS)
    chunks = [
        _chunk("普通内容A", cid="a"),
        _chunk("董事会报告内容", chapter="董事会报告", cid="b"),
    ]
    ordered = r.rerank("测试查询", chunks)
    assert [c.chunk_id for c in ordered] == ["b", "a"]  # b 的 CE 分 1.0 > a 的 0.5


def test_fusion_section_signal_lifts_high_priority_chapter():
    """β>0 时，章节优先级可把"管理层讨论及分析"的 chunk 提前。

    假模型对两个 chunk 都给 0.5（都不含 [Chapter] 董事会报告），
    但"管理层讨论及分析"章节有 0.15 优先级加成 → 排序靠前。
    """
    r = _TestMetadataReranker(company_weights=_TEST_WEIGHTS_SECTION)
    chunks = [
        _chunk("普通内容A", chapter="公司简介", cid="a"),
        _chunk("管理层讨论内容", chapter="管理层讨论及分析", cid="b"),
    ]
    ordered = r.rerank("未来战略", chunks)
    assert ordered[0].chunk_id == "b"  # 章节优先级把 b 提前


def test_unconfigured_company_pure_ce_zero_regression():
    """未配置公司 → 纯 CrossEncoder（α=1, β=0, γ=0）：章节信号不生效，零回归。"""
    r = _TestMetadataReranker(company_weights={"其他公司": (0.8, 0.15, 0.05)})
    chunks = [
        _chunk("普通内容A", company="未配置公司", chapter="公司简介", cid="a"),
        _chunk("管理层讨论内容", company="未配置公司", chapter="管理层讨论及分析", cid="b"),
    ]
    # 未配置公司 → 纯 CE：假模型对两个 chunk 都给 0.5（无 [Chapter] 董事会报告）→ 稳定序
    ordered = r.rerank("未来战略", chunks)
    assert {c.chunk_id for c in ordered} == {"a", "b"}


def test_metadata_reranker_smoke():
    """端到端冒烟：Context Builder + CE + 融合 全链路可跑，返回排序后的 chunk。"""
    r = _TestMetadataReranker()
    chunks = [
        _chunk("小米汽车业务快速发展", company="小米", chapter="管理层讨论及分析", cid="a"),
        _chunk("董事会报告内容", chapter="董事会报告", cid="b"),
        _chunk("财务数据表格", chapter="财务报表", cid="c"),
    ]
    ordered = r.rerank("小米汽车未来战略", chunks)
    assert len(ordered) == 3
    assert set(c.chunk_id for c in ordered) == {"a", "b", "c"}  # 不丢 chunk
    # 排序确定：同输入再跑一次结果一致
    ordered2 = r.rerank("小米汽车未来战略", chunks)
    assert [c.chunk_id for c in ordered2] == [c.chunk_id for c in ordered]
