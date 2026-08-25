"""Document 原始模型 + DocumentMetadata 统一规范 + 元数据链路测试。"""

from app.rag import load_documents, split_documents
from app.rag.document import Document, DocumentChunk, DocumentMetadata

_METADATA_FIELDS = (
    "source",
    "source_name",
    "company",
    "doc_type",
    "page",
    "section",
    "chapter",
    "title",
    "table",
    "header",
    "footer",
    "original_text",
    "created_time",
)


def test_document_is_raw_before_split(tmp_path):
    """loader 输出应为原始 Document，而非已切分的 DocumentChunk。"""
    md = tmp_path / "基金规则.md"
    md.write_text("# 基金规则\n\n## 赎回\n\nT+1到账", encoding="utf-8")

    docs = load_documents(str(md))

    assert isinstance(docs[0], Document)
    assert not isinstance(docs[0], DocumentChunk)


def test_metadata_has_uniform_fields_for_all_loaders(tmp_path):
    """markdown 与 txt loader 应输出同一组固定字段（防字段漂移）。"""
    md = tmp_path / "基金规则.md"
    md.write_text("# 基金规则\n\n赎回T+1到账。", encoding="utf-8")
    txt = tmp_path / "行业报告.txt"
    txt.write_text("动力电池行业装机量持续增长。", encoding="utf-8")

    for f in (md, txt):
        doc = load_documents(str(f))[0]
        assert isinstance(doc.metadata, DocumentMetadata)
        for field in _METADATA_FIELDS:
            assert hasattr(doc.metadata, field), f"缺失字段: {field}"


def test_markdown_metadata_flows_to_chunk_with_section(tmp_path):
    """# 基金规则 / ## 赎回 / T+1到账 → chunk 应带 section 章节路径。"""
    md = tmp_path / "基金规则.md"
    md.write_text("# 基金规则\n\n## 赎回\n\nT+1到账", encoding="utf-8")

    docs = load_documents(str(md))
    docs[0].metadata.company = "测试公司"
    chunks = split_documents(docs)

    assert len(chunks) >= 1
    for c in chunks:
        assert c.company == "测试公司"
        assert c.metadata["doc_type"] == "markdown"
    # 标题路径写入 source_name 与 section
    assert any("赎回" in c.source_name for c in chunks)
    assert any("赎回" in c.metadata.get("section", "") for c in chunks)
    assert any("T+1到账" in c.text for c in chunks)


def test_company_filled_via_metadata(tmp_path):
    """company 通过 DocumentMetadata 注入，splitter 提取到各 chunk。"""
    txt = tmp_path / "资料.txt"
    txt.write_text("内容一。\n\n内容二。", encoding="utf-8")

    docs = load_documents(str(txt))
    docs[0].metadata.company = "宁德时代"

    chunks = split_documents(docs)
    assert all(c.company == "宁德时代" for c in chunks)
