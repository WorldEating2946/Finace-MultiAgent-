"""多源知识融合测试（PR #35）。

覆盖：SourceType / EnterpriseKnowledgePackage schema、ConflictDetector 跨源冲突检测、
SourceFusion 融合（证据合并 / 冲突保留 / 单源保留）、source_type 数据通道
（ProfileExtractor 过滤 + EvidenceRef 标注）。
用 mock 数据隔离真实向量库与 LLM。
"""

from app.rag.document import DocumentChunk
from app.rag.profile.extractor import ProfileExtractor
from app.rag.profile.schema import CompanyProfile, EvidenceRef, ProfileItem
from app.rag.source.conflict import ConflictDetector
from app.rag.source.fusion import SourceFusion
from app.rag.source.schema import (
    FUSION_DIMENSIONS,
    EnterpriseKnowledgePackage,
    SourceConflict,
    SourceType,
)


# ── Schema ─────────────────────────────────────────────────────

def test_source_type_enum_values():
    """SourceType 枚举值与文档语义类型一致。"""
    assert SourceType.ANNUAL_REPORT.value == "annual_report"
    assert SourceType.RESEARCH_REPORT.value == "research_report"
    assert SourceType.POLICY.value == "policy"
    assert SourceType.NEWS.value == "news"
    # str 枚举可直接用于源名比较
    assert "annual_report" in set(s.value for s in SourceType)


def test_package_schema_structure():
    """EnterpriseKnowledgePackage 结构化正确。"""
    pkg = EnterpriseKnowledgePackage(
        company_name="小米",
        profiles={"annual_report": CompanyProfile(company_name="小米")},
        fused=CompanyProfile(company_name="小米"),
        conflicts=[],
        evidence_summary={"annual_report": 3},
    )
    assert pkg.company_name == "小米"
    assert pkg.conflicts == []
    assert pkg.evidence_summary["annual_report"] == 3


# ── 冲突检测 ───────────────────────────────────────────────────

def _item(name: str, desc: str, source_type: str = "") -> ProfileItem:
    return ProfileItem(
        name=name,
        description=desc,
        evidence=[EvidenceRef(source="x.pdf", source_type=source_type, page=1, quote=desc)],
    )


def _profile(items: list[ProfileItem]) -> CompanyProfile:
    p = CompanyProfile(company_name="小米")
    p.business_segments = items
    return p


def test_conflict_detects_contradictory_descriptions():
    """同名实体、矛盾描述 → 冲突。"""
    ar = _profile([_item("智能电动汽车业务", "公司持续推进智能电动汽车业务", "annual_report")])
    rr = _profile([_item("汽车业务", "汽车业务短期亏损压力较大", "research_report")])
    conflicts = ConflictDetector().detect({"annual_report": ar, "research_report": rr})

    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.dimension == "business_segments"
    assert c.source_a == "annual_report" and c.source_b == "research_report"
    assert c.evidence_a is not None and c.evidence_b is not None


def test_conflict_no_conflict_when_similar():
    """同名实体、相似描述 → 无冲突。"""
    ar = _profile([_item("智能手机", "智能手机业务为全球前三", "annual_report")])
    rr = _profile([_item("智能手机", "智能手机出货量全球前三", "research_report")])
    conflicts = ConflictDetector().detect({"annual_report": ar, "research_report": rr})
    assert conflicts == []


def test_conflict_different_entities_no_conflict():
    """不同实体（无命名关联）→ 无冲突。"""
    ar = _profile([_item("智能手机", "核心主业", "annual_report")])
    rr = _profile([_item("新能源汽车", "新增长引擎", "research_report")])
    conflicts = ConflictDetector().detect({"annual_report": ar, "research_report": rr})
    assert conflicts == []


def test_conflict_empty_profiles():
    """空画像 → 无冲突。"""
    assert ConflictDetector().detect({}) == []
    ar = _profile([])
    rr = _profile([])
    assert ConflictDetector().detect({"annual_report": ar, "research_report": rr}) == []


# ── 融合 ───────────────────────────────────────────────────────

def _builder_for(profiles: dict) -> callable:
    """测试 seam：按 source_type 返回预设画像。"""
    def _build(company: str, source_type: str) -> CompanyProfile:
        return profiles[source_type]
    return _build


def test_fusion_merges_same_entity_evidence():
    """同名实体跨源 → 证据合并（evidence 来自两个源）。"""
    ar = _profile([_item("智能手机", "智能手机业务全球前三", "annual_report")])
    rr = _profile([_item("智能手机", "智能手机出货量全球前三", "research_report")])
    fusion = SourceFusion(
        "小米",
        sources=["annual_report", "research_report"],
        _profile_builder=_builder_for({"annual_report": ar, "research_report": rr}),
    )
    pkg = fusion.build()

    # 融合画像中"智能手机"证据数 = 年报 1 + 研报 1 = 2
    segs = pkg.fused.business_segments
    assert len(segs) == 1
    assert segs[0].name == "智能手机"
    assert len(segs[0].evidence) == 2
    types = {e.source_type for e in segs[0].evidence}
    assert types == {"annual_report", "research_report"}
    # 无冲突
    assert pkg.conflicts == []
    # evidence_summary 按源统计
    assert pkg.evidence_summary == {"annual_report": 1, "research_report": 1}


def test_fusion_conflict_keeps_annual_report():
    """冲突实体 → fused 保留年报版本，conflicts 记录差异。"""
    ar = _profile([_item("智能电动汽车业务", "公司持续推进智能电动汽车业务", "annual_report")])
    rr = _profile([_item("汽车业务", "汽车业务短期亏损压力较大", "research_report")])
    fusion = SourceFusion(
        "小米",
        sources=["annual_report", "research_report"],
        _profile_builder=_builder_for({"annual_report": ar, "research_report": rr}),
    )
    pkg = fusion.build()

    assert len(pkg.conflicts) == 1
    # fused 保留年报命名与描述
    seg = pkg.fused.business_segments[0]
    assert seg.name == "智能电动汽车业务"
    assert "持续推进" in seg.description


def test_fusion_preserves_single_source_entity():
    """单源实体 → 原样保留。"""
    ar = _profile([_item("AIoT", "生态链产品", "annual_report")])
    rr = _profile([])
    fusion = SourceFusion(
        "小米",
        sources=["annual_report", "research_report"],
        _profile_builder=_builder_for({"annual_report": ar, "research_report": rr}),
    )
    pkg = fusion.build()
    assert len(pkg.fused.business_segments) == 1
    assert pkg.fused.business_segments[0].name == "AIoT"


def test_fusion_fused_industry_prefers_annual_report():
    """fused.industry 优先年报。"""
    ar = CompanyProfile(company_name="小米", industry="智能硬件与消费电子", business_segments=[])
    rr = CompanyProfile(company_name="小米", industry="科技制造", business_segments=[])
    fusion = SourceFusion(
        "小米",
        sources=["annual_report", "research_report"],
        _profile_builder=_builder_for({"annual_report": ar, "research_report": rr}),
    )
    pkg = fusion.build()
    assert pkg.fused.industry == "智能硬件与消费电子"


# ── source_type 数据通道 ───────────────────────────────────────

def _chunk(text: str, *, source_type: str = "", cid: str = "c0") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid, company="小米", doc_type="pdf",
        source="小米集团2025年报.pdf", source_name="小米集团2025年报",
        page=19, text=text,
        metadata={"source_type": source_type},
    )


def test_extractor_filters_by_source_type():
    """ProfileExtractor source_type 过滤：只保留指定源 chunk。"""
    chunks = [
        _chunk("年报文段", source_type="annual_report", cid="a"),
        _chunk("研报文段", source_type="research_report", cid="b"),
        _chunk("政策文段", source_type="policy", cid="c"),
    ]
    ex = ProfileExtractor("小米", top_k=2, source_type="research_report")
    filtered = ex._filter_chunks(chunks)
    assert len(filtered) == 1
    assert filtered[0].chunk_id == "b"


def test_extractor_no_source_type_passthrough():
    """source_type 空串 = 不过滤（向后兼容），截取 top_k。"""
    chunks = [
        _chunk("文段1", source_type="annual_report", cid="a"),
        _chunk("文段2", source_type="annual_report", cid="b"),
        _chunk("文段3", source_type="policy", cid="c"),
    ]
    ex = ProfileExtractor("小米", top_k=2)
    filtered = ex._filter_chunks(chunks)
    assert len(filtered) == 2  # 未过滤，仅截取 top_k
    assert filtered[0].chunk_id == "a"


def test_format_chunks_includes_source_type_in_ref_map():
    """_format_chunks 把 source_type 写入 ref_map（证据归因可信字段）。"""
    ex = ProfileExtractor("小米", top_k=3)
    chunks = [_chunk("智能手机业务", source_type="annual_report", cid="a")]
    _, ref_map = ex._format_chunks(chunks)
    assert ref_map[0]["source_type"] == "annual_report"


def test_parse_field_output_writes_source_type():
    """_parse_field_output 把 source_type 写入 EvidenceRef。"""
    ex = ProfileExtractor("小米", top_k=3)
    chunks = [_chunk("智能手机业务", source_type="research_report", cid="a")]
    _, ref_map = ex._format_chunks(chunks)
    llm_raw = '[{"name":"智能手机","description":"主业","evidence_refs":[0],"quotes":["智能手机业务"]}]'
    items = ex._parse_field_output(llm_raw, ref_map)
    assert items[0].evidence[0].source_type == "research_report"
