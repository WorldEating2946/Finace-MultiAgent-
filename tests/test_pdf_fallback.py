"""PDF 多解析器 fallback + 质量门禁 单元测试（monkeypatch 模拟解析器）。"""

import pytest

from app.rag.document import Document
from app.rag.loaders.pdf.types import PageText
from app.rag.loaders.pdf_loader import PDFLoader, PDFQualityError

_GOOD = [PageText(1, "小米集团主要业务包括智能手机、IoT与生活消费产品。")]
_GARBAGE = [PageText(1, "Ṗ⸶屣⊛㥄奃 㛮⸶⸳⟳␌灭劳㕉⎌ᷯ")]


class _Parser:
    def __init__(self, name, pages):
        self.name = name
        self._pages = pages

    def parse(self, file_path):
        return self._pages


def _make_pdf(tmp_path, name="样本.pdf") -> str:
    f = tmp_path / name
    f.write_bytes(b"%PDF-1.4 fake content")
    return str(f)


def test_loader_uses_first_parser_passing_quality(monkeypatch, tmp_path):
    """主解析器质量达标 → 直接采用，不触发 fallback。"""
    loader = PDFLoader()
    monkeypatch.setattr(loader, "_parsers", (_Parser("pymupdf", _GOOD),))
    docs = loader.load(_make_pdf(tmp_path))
    assert isinstance(docs[0], Document)
    assert "智能手机" in docs[0].text
    assert docs[0].metadata.page == 1
    assert docs[0].metadata.doc_type == "pdf"


def test_loader_falls_back_when_first_low_quality(monkeypatch, tmp_path):
    """主解析器低质量 → 回退到 fallback 解析器（质量达标则采用）。"""
    loader = PDFLoader()
    monkeypatch.setattr(
        loader, "_parsers", (_Parser("pymupdf", _GARBAGE), _Parser("pdfplumber", _GOOD))
    )
    docs = loader.load(_make_pdf(tmp_path))
    assert "智能手机" in docs[0].text  # 来自 fallback 解析器


def test_loader_blocks_when_all_low_quality(monkeypatch, tmp_path):
    """所有解析器质量都低 → 抛 PDFQualityError 阻断（防垃圾进 embedding）。"""
    loader = PDFLoader()
    monkeypatch.setattr(loader, "_parsers", (_Parser("pymupdf", _GARBAGE), _Parser("pdfplumber", _GARBAGE)))
    with pytest.raises(PDFQualityError):
        loader.load(_make_pdf(tmp_path))


def test_loader_skips_failed_parser(monkeypatch, tmp_path):
    """解析器抛异常（缺依赖等）→ 跳过并尝试下一个。"""
    class CrashParser:
        name = "crash"

        def parse(self, file_path):
            raise ImportError("缺依赖")

    loader = PDFLoader()
    monkeypatch.setattr(loader, "_parsers", (CrashParser(), _Parser("pdfplumber", _GOOD)))
    docs = loader.load(_make_pdf(tmp_path))
    assert "智能手机" in docs[0].text
