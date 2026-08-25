"""PDF 多解析器策略（fallback 链）。

    PDF → PyMuPDF → 质量检查 → ≥0.8 → 继续
                           ↓ <0.8
                      pdfplumber → 质量检查 → ≥0.8 → 继续
                           ↓ <0.8
                        OCR（预留，未实现）→ 阻断（PDFQualityError）

质量门禁防止文本层损坏（乱码）的 PDF 静默进入 embedding。
"""

from app.rag.loaders.pdf.types import PageText

__all__ = ["PageText"]
