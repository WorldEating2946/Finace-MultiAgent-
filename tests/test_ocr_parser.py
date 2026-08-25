"""OCR 解析器单元测试。

需要 Tesseract 二进制 + chi_sim 语言包（未安装则跳过）。
OCR 较慢（~5-10s/页），仅单页验证。
"""

import pytest

pytest.importorskip("pytesseract")


def _make_ocr_pdf(tmp_path):
    """生成含大字中文的 PDF（Tesseract 可识别）。"""
    import fitz

    path = tmp_path / "ocr_test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 200), "宁德时代动力电池出货量全球第一", fontname="china-s", fontsize=30)
    page.insert_text((72, 250), "营业收入同比增长20%", fontname="china-s", fontsize=30)
    doc.save(str(path))
    doc.close()
    return path


@pytest.mark.real
def test_ocr_parser_recovers_chinese_text(tmp_path):
    """OCR 应从渲染图像中恢复中文文本。"""
    from app.rag.loaders.pdf.ocr_parser import OcrParser

    pdf = _make_ocr_pdf(tmp_path)
    try:
        pages = OcrParser().parse(str(pdf))
    except ImportError:
        pytest.skip("Tesseract 不可用（需安装 tesseract + chi_sim）")

    assert len(pages) >= 1
    text = pages[0].text
    assert "宁德时代" in text, f"OCR 未恢复中文: {text[:80]!r}"
    assert "动力电池" in text


def test_ocr_parser_in_fallback_chain():
    """OCR 解析器应在 pdf_loader fallback 链中（配置开启时）。"""
    from app.core.config import settings
    from app.rag.loaders.pdf.ocr_parser import OcrParser
    from app.rag.loaders.pdf_loader import PDFLoader

    assert settings.rag_ocr is True
    loader = PDFLoader()
    assert any(isinstance(p, OcrParser) for p in loader._parsers)
