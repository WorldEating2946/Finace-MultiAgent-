"""PyMuPDF（fitz）PDF 文本解析器（主解析器）。"""

from __future__ import annotations

from app.rag.loaders.pdf.types import PageText


class PymupdfParser:
    """基于 PyMuPDF 的逐页文本抽取。"""

    name = "pymupdf"

    def parse(self, file_path: str) -> list[PageText]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF 解析需要 pymupdf，请执行：uv pip install -r requirements.txt"
            )

        pages: list[PageText] = []
        pdf = fitz.open(file_path)
        try:
            for page in pdf:
                raw = page.get_text()
                if not raw.strip():
                    continue  # 无文本层（扫描件）跳过
                pages.append(PageText(page=page.number + 1, text=raw))
        finally:
            pdf.close()

        return pages
