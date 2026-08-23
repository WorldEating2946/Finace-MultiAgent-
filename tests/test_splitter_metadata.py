"""Splitter 策略分发 + 元数据提取单元测试（Document → DocumentChunk）。"""

from app.rag.document import Document
from app.rag.splitter import split_documents


def _make_doc(
    text: str,
    *,
    doc_type: str = "text",
    source_name: str = "测试文档",
    company: str = "宁德时代",
    source: str = "/tmp/测试文档.md",
) -> Document:
    """构建 Document（模拟 loader 输出 + 入库填充 company）。"""
    return Document(
        text=text,
        metadata={
            "doc_type": doc_type,
            "source": source,
            "source_name": source_name,
            "page": 0,
            "company": company,
        },
    )


def test_plain_text_dispatch_and_metadata_preserved():
    doc = _make_doc("动力电池行业装机量持续增长。")

    result = split_documents([doc])

    assert len(result) == 1
    c = result[0]
    assert c.doc_type == "text"
    assert c.company == "宁德时代"
    assert c.source_name == "测试文档"
    assert c.metadata["chunk_index"] == 0


def test_short_text_keeps_single_chunk_and_inherits_metadata():
    doc = _make_doc("宁德时代是全球动力电池龙头。", source_name="企业介绍")

    result = split_documents([doc])

    assert len(result) == 1
    c = result[0]
    assert c.text == "宁德时代是全球动力电池龙头。"
    assert c.company == "宁德时代"
    assert c.source_name == "企业介绍"
    assert c.source == "/tmp/测试文档.md"
    assert c.metadata["chunk_index"] == 0


def test_markdown_dispatch_uses_header_path():
    md = (
        "# 第一章 公司概况\n\n宁德时代是全球动力电池龙头。\n\n"
        "## 1.1 商业模式\n\n主要依靠动力电池系统销售。"
    )
    doc = _make_doc(md, doc_type="markdown", source_name="招股书")

    result = split_documents([doc])

    names = [c.source_name for c in result]
    assert any("第一章" in n for n in names)
    assert any("1.1 商业模式" in n for n in names)
    # 章节路径同时写入 chunk.metadata["section"]
    assert any("1.1 商业模式" in c.metadata.get("section", "") for c in result)


def test_long_plain_text_splits_into_multiple_chunks():
    doc = _make_doc("动力电池行业装机量持续增长。" * 100)

    result = split_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(result) > 1
    for c in result:
        assert len(c.text) <= 220  # chunk_size + overlap 上限
        assert c.text.strip() != ""


def test_chunk_id_stable_for_same_content():
    doc_a = _make_doc("宁德时代是全球动力电池龙头。" * 100)
    doc_b = _make_doc("宁德时代是全球动力电池龙头。" * 100)

    r_a = split_documents([doc_a], chunk_size=200, chunk_overlap=20)
    r_b = split_documents([doc_b], chunk_size=200, chunk_overlap=20)

    assert [c.chunk_id for c in r_a] == [c.chunk_id for c in r_b]


def test_chunk_index_sequence():
    doc = _make_doc("动力电池行业持续增长。" * 100)

    result = split_documents([doc], chunk_size=200, chunk_overlap=20)

    indexes = [c.metadata["chunk_index"] for c in result]
    assert indexes == list(range(len(result)))


def test_plain_markdown_without_headers_splits_by_size():
    # 无标题的 markdown 视为纯文本处理，仅按 chunk_size 切分
    doc = _make_doc("宁德时代主营业务。" * 200, doc_type="markdown")

    result = split_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(result) > 1
    assert all(c.source_name == "测试文档" for c in result)
