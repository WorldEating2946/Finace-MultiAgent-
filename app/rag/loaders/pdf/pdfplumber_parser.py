"""pdfplumber PDF 文本解析器（fallback）。

部分 PDF 生成工具对 PyMuPDF 的 ToUnicode 处理不一致，
pdfplumber（pdfminer.six 布局分析）可能得到不同结果。
"""

from __future__ import annotations

from app.rag.loaders.pdf.types import PageText


class PdfplumberParser:
    """基于 pdfplumber 的逐页文本抽取。"""

    name = "pdfplumber"

    def parse(self, file_path: str) -> list[PageText]:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber 解析需要 pdfplumber，请执行：uv pip install -r requirements.txt"
            )

        pages: list[PageText] = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                if not raw.strip():
                    continue
                pages.append(PageText(page=i, text=raw))

        return pages
