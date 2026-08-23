"""企业知识画像测试（PR #34）。

覆盖：schema / 证据归因（LLM 索引 → 补全 source/page）/ 越界过滤 / build_profile / storage。
用 mock LLM 调用（_call seam）与 mock extractor 隔离网络与真实向量库。
"""

from pathlib import Path

from app.rag.document import DocumentChunk
from app.rag.profile.extractor import ProfileExtractor, build_profile
from app.rag.profile.schema import CompanyProfile, EvidenceRef, ProfileItem
from app.rag.profile.storage import load_profile, save_profile


def _chunk(text: str, *, chapter: str = "管理层讨论及分析", page: int = 19, cid: str = "c0") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid, company="小米", doc_type="pdf",
        source="小米集团2025年报.pdf", source_name="小米集团2025年报",
        page=page, text=text,
        metadata={"chapter": chapter},
    )


# ── Schema ─────────────────────────────────────────────────────

def test_schema_evidence_structure():
    """CompanyProfile + ProfileItem + EvidenceRef 结构化正确。"""
    profile = CompanyProfile(
        company_name="小米",
        industry="智能硬件与消费电子",
        business_segments=[
            ProfileItem(
                name="智能手机",
                description="核心主业",
                evidence=[EvidenceRef(source="年报", chapter="管理层讨论及分析", page=19, quote="2025年智能手机收入...")],
            )
        ],
    )
    seg = profile.business_segments[0]
    assert seg.name == "智能手机"
    assert seg.evidence[0].page == 19
    assert seg.evidence[0].quote


# ── 证据归因 ───────────────────────────────────────────────────

def _extractor() -> ProfileExtractor:
    return ProfileExtractor("小米", top_k=3)


def test_format_chunks_builds_ref_map():
    """LLM 只看到索引号，source/page 在 ref_map（不可伪造）。"""
    chunks = [
        _chunk("小米智能手机业务快速发展", chapter="管理层讨论及分析", page=19, cid="a"),
        _chunk("小米智能电动汽车业务", chapter="董事会报告", page=57, cid="b"),
    ]
    ex = _extractor()
    context, ref_map = ex._format_chunks(chunks)

    assert "[0] 章节:管理层讨论及分析 页码:19" in context
    assert "[1] 章节:董事会报告 页码:57" in context
    assert ref_map[0]["source"] == "小米集团2025年报.pdf"
    assert ref_map[0]["page"] == 19
    assert ref_map[1]["chapter"] == "董事会报告"


def test_parse_field_output_resolves_evidence():
    """LLM 返回 evidence_refs 索引 → 后处理补全 source/page/chapter。"""
    ex = _extractor()
    chunks = [
        _chunk("小米智能手机业务快速发展", chapter="管理层讨论及分析", page=19, cid="a"),
        _chunk("小米智能电动汽车业务", chapter="董事会报告", page=57, cid="b"),
    ]
    _, ref_map = ex._format_chunks(chunks)
    llm_raw = (
        '[{"name":"智能手机","description":"核心主业","evidence_refs":[0],'
        '"quotes":["小米智能手机业务快速发展"]}]'
    )

    items = ex._parse_field_output(llm_raw, ref_map)

    assert len(items) == 1
    assert items[0].name == "智能手机"
    assert items[0].evidence[0].source == "小米集团2025年报.pdf"
    assert items[0].evidence[0].chapter == "管理层讨论及分析"
    assert items[0].evidence[0].page == 19
    assert items[0].evidence[0].quote == "小米智能手机业务快速发展"


def test_parse_field_output_filters_out_of_range_refs():
    """越界 evidence_refs（LLM 幻觉）被静默丢弃；无有效证据的实体丢弃。"""
    ex = _extractor()
    chunks = [_chunk("小米智能手机业务", cid="a")]
    _, ref_map = ex._format_chunks(chunks)  # 只有 chunk 0

    # refs 含 5（越界）→ 该实体无有效证据 → 丢弃
    llm_raw = '[{"name":"编造的实体","description":"不存在","evidence_refs":[5]}]'
    items = ex._parse_field_output(llm_raw, ref_map)
    assert items == []

    # 部分有效：refs [0, 9] → 只保留 0
    llm_raw2 = '[{"name":"智能手机","description":"主业","evidence_refs":[0,9],"quotes":["内容","x"]}]'
    items2 = ex._parse_field_output(llm_raw2, ref_map)
    assert len(items2) == 1
    assert len(items2[0].evidence) == 1  # 越界 ref 被过滤


def test_parse_field_output_json_codeblock():
    """LLM 输出 ```json``` 包裹 → 仍能解析。"""
    ex = _extractor()
    chunks = [_chunk("小米智能手机业务", cid="a")]
    _, ref_map = ex._format_chunks(chunks)
    llm_raw = '```json\n[{"name":"智能手机","description":"主业","evidence_refs":[0],"quotes":["内容"]}]\n```'
    items = ex._parse_field_output(llm_raw, ref_map)
    assert len(items) == 1


# ── build_profile ──────────────────────────────────────────────

class _MockExtractor:
    """mock extractor：每字段返回一个带证据的实体。"""

    def extract_field(self, field_name: str, retrieval_query: str) -> list[ProfileItem]:
        if field_name == "industry":
            return [ProfileItem(name="行业", description="智能硬件与消费电子")]
        return [
            ProfileItem(
                name=f"{field_name}_实体",
                description=f"{field_name}描述",
                evidence=[EvidenceRef(source="年报", chapter="管理层讨论及分析", page=19, quote="q")],
            )
        ]


def test_build_profile_assembles_all_fields():
    """build_profile 组装全部画像维度。"""
    profile = build_profile("小米", extractor=_MockExtractor())

    assert profile.company_name == "小米"
    assert profile.industry == "智能硬件与消费电子"
    for dim in (
        "business_segments", "products", "technologies", "customers",
        "geographic_markets", "competitive_advantages", "risks", "strategic_direction",
    ):
        items = getattr(profile, dim)
        assert items, f"维度 {dim} 为空"
        assert items[0].evidence  # 每字段必须带证据链


# ── storage ────────────────────────────────────────────────────

def test_storage_roundtrip(tmp_path):
    """save → load 往返，证据链完整。"""
    profile = CompanyProfile(
        company_name="测试公司",
        business_segments=[ProfileItem(name="主业", evidence=[EvidenceRef(source="s", page=1, quote="q")])],
    )
    path = save_profile(profile, path=str(tmp_path / "测试公司.json"))
    loaded = load_profile("测试公司", path=str(path))

    assert loaded is not None
    assert loaded.company_name == "测试公司"
    assert loaded.business_segments[0].evidence[0].page == 1


def test_load_profile_missing_returns_none(tmp_path):
    assert load_profile("不存在的公司", path=str(tmp_path / "x.json")) is None
