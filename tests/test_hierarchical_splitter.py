"""层级化切分（Structure-aware）单元测试。"""

from app.rag.loaders.pdf_loader import PDFLoader
from app.rag.splitters.hierarchical_splitter import split_pdf_hierarchical


def _load_pdf_docs(xiaomi_pdf_path):
    docs = PDFLoader().load(str(xiaomi_pdf_path))
    docs[0].metadata.company = "小米"
    return docs


def test_pdf_chunks_carry_section_path(xiaomi_pdf_path):
    """chunk 应携带章节路径（如 "第四章 业务回顾 > 4.1 智能手机业务"）。"""
    docs = _load_pdf_docs(xiaomi_pdf_path)
    chunks = split_pdf_hierarchical(docs, chunk_size=512, chunk_overlap=100)

    sectioned = [c for c in chunks if c.metadata["section"]]
    assert len(sectioned) > 0
    assert any("4.1" in c.metadata["section"] for c in sectioned)
    assert any("第三章" in c.metadata["section"] for c in sectioned)


def test_pdf_chunks_carry_chapter(xiaomi_pdf_path):
    """chunk 应携带章标题（metadata.chapter）。"""
    docs = _load_pdf_docs(xiaomi_pdf_path)
    chunks = split_pdf_hierarchical(docs, chunk_size=512, chunk_overlap=100)

    chapters = {c.metadata["chapter"] for c in chunks if c.metadata["chapter"]}
    assert any("第四章 业务回顾" in c for c in chapters)
    assert any("第五章 财务报告" in c for c in chapters)


def test_pdf_chunks_traceable(xiaomi_pdf_path):
    """chunk 应带页码与来源（可追溯）。"""
    docs = _load_pdf_docs(xiaomi_pdf_path)
    chunks = split_pdf_hierarchical(docs, chunk_size=512, chunk_overlap=100)

    assert len(chunks) > 0
    assert all(c.page is not None for c in chunks)
    assert all(c.metadata["source"] for c in chunks)
    assert all(c.doc_type == "pdf" for c in chunks)
    assert all(c.company == "小米" for c in chunks)


def test_every_section_has_titled_chunk(xiaomi_pdf_path):
    """每个识别到的节，至少一个 chunk 的文本保留节标题（LLM 章节上下文）。"""
    docs = _load_pdf_docs(xiaomi_pdf_path)
    chunks = split_pdf_hierarchical(docs, chunk_size=512, chunk_overlap=100)

    sections = {c.metadata["section"] for c in chunks if c.metadata["section"]}
    titled = {
        c.metadata["section"]
        for c in chunks
        if c.metadata["section"] and c.metadata["section"].split(" > ")[-1] in c.text
    }
    assert titled == sections  # 每个节都至少有一个 chunk 含节标题
