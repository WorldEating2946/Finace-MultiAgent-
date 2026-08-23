"""Markdown Loader 单元测试（Document 输出与元数据）。"""

import pytest

from app.rag.document import Document
from app.rag.loaders.markdown_loader import MarkdownLoader


def test_markdown_loader_returns_document_with_metadata(tmp_path):
    md = tmp_path / "基金规则.md"
    md.write_text("# 基金规则\n\n## 赎回\n\nT+1到账", encoding="utf-8")

    docs = MarkdownLoader().load(str(md))

    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert "T+1到账" in doc.text
    assert doc.metadata.doc_type == "markdown"
    assert doc.metadata.source.endswith("基金规则.md")
    assert doc.metadata.source_name == "基金规则"
    assert doc.metadata.page is None  # 无分页文档
    assert doc.metadata.created_time  # 取文件 mtime，非空


def test_markdown_loader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MarkdownLoader().load(str(tmp_path / "不存在.md"))
