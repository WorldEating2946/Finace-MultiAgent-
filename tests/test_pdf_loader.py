"""PDF Loader 单元测试。"""

import pytest

from app.rag import load_documents
from app.rag.loaders import get_loader
from app.rag.loaders.pdf_loader import PDFLoader


def test_get_loader_dispatches_pdf():
    assert isinstance(get_loader("x.pdf"), PDFLoader)


def test_pdf_load_many_pages(xiaomi_pdf_path):
    """PDF 应逐页解析为 Document，页码从 1 开始。"""
    docs = load_documents(str(xiaomi_pdf_path))

    assert len(docs) > 200
    assert docs[0].metadata.page == 1
    assert docs[0].metadata.doc_type == "pdf"
    assert docs[0].metadata.source.endswith(".pdf")
    assert docs[0].metadata.section == ""  # v1 不解析 section


def test_pdf_metadata_uniform(xiaomi_pdf_path):
    """PDF 输出 DocumentMetadata 统一字段（与 markdown/txt 一致）。"""
    doc = load_documents(str(xiaomi_pdf_path))[0]
    for field in (
        "source",
        "source_name",
        "company",
        "doc_type",
        "page",
        "section",
        "created_time",
    ):
        assert hasattr(doc.metadata, field), f"缺失字段: {field}"


def test_pdf_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PDFLoader().load(str(tmp_path / "不存在.pdf"))
