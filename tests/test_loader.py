"""Document Loader 统一入口单元测试（文件类型分发 → list[Document]）。"""

import pytest

from app.rag import load_documents
from app.rag.document import Document
from app.rag.loaders import get_loader
from app.rag.loaders.markdown_loader import MarkdownLoader
from app.rag.loaders.txt_loader import TXTLoader


def test_load_markdown_file_returns_document(tmp_path):
    md_file = tmp_path / "企业介绍.md"
    md_file.write_text("# 宁德时代\n\n宁德时代是全球动力电池龙头。", encoding="utf-8")

    docs = load_documents(str(md_file))

    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert "宁德时代是全球动力电池龙头" in doc.text
    assert doc.metadata.doc_type == "markdown"
    assert doc.metadata.source_name == "企业介绍"
    assert doc.metadata.source.endswith(".md")


def test_load_txt_file_returns_document(tmp_path):
    txt_file = tmp_path / "行业报告.txt"
    txt_file.write_text("动力电池行业装机量持续增长。", encoding="utf-8")

    docs = load_documents(str(txt_file))

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].metadata.doc_type == "text"
    assert "动力电池行业装机量持续增长" in docs[0].text


def test_get_loader_dispatches_by_extension():
    assert isinstance(get_loader("x.md"), MarkdownLoader)
    assert isinstance(get_loader("x.markdown"), MarkdownLoader)
    assert isinstance(get_loader("x.txt"), TXTLoader)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(str(tmp_path / "不存在.md"))


def test_unsupported_extension_raises(tmp_path):
    docx_file = tmp_path / "招股书.docx"
    docx_file.write_bytes(b"PK fake docx")

    with pytest.raises(ValueError):
        load_documents(str(docx_file))


def test_directory_raises(tmp_path):
    with pytest.raises(ValueError):
        load_documents(str(tmp_path))
