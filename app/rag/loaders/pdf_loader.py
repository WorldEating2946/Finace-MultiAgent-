"""PDF 文档加载器（多解析器 fallback + 质量门禁）。

策略：
    PDF → PyMuPDF → 质量检查 → ≥0.8 → 继续
                       ↓ <0.8
                  pdfplumber → 质量检查 → ≥0.8 → 继续
                       ↓ <0.8
                     OCR（预留，未实现）→ 阻断（PDFQualityError）

文本层损坏（乱码，如 ToUnicode 映射错误）的 PDF 会被阻断，
避免"无报错但结果错误"的垃圾文本进入 embedding。

metadata（DocumentMetadata 统一规范）：
    source / source_name / doc_type="pdf" / page=N / original_text / created_time
"""

import logging
from pathlib import Path

from app.core.config import settings
from app.rag.document import Document, DocumentMetadata
from app.rag.loaders.base import DocumentLoader, file_created_time
from app.rag.loaders.pdf.ocr_parser import OcrParser
from app.rag.loaders.pdf.pdfplumber_parser import PdfplumberParser
from app.rag.loaders.pdf.pymupdf_parser import PymupdfParser
from app.rag.loaders.pdf.quality_checker import QUALITY_THRESHOLD, quality_score
from app.rag.loaders.pdf.types import PageText
from app.rag.parsers.table_parser import TableData, TableParser
from app.rag.parsers.text_normalizer import normalize_text

logger = logging.getLogger(__name__)


class PDFQualityError(ValueError):
    """PDF 文本层损坏或不可解析（质量低于阈值，无法安全入库）。"""


class PDFLoader(DocumentLoader):
    """加载 .pdf 文件为逐页 Document；多解析器 fallback + 质量门禁。"""

    def __init__(self) -> None:
        # fallback 链：PyMuPDF → pdfplumber → OCR（文本层损坏时的最后手段）
        parsers: list = [PymupdfParser(), PdfplumberParser()]
        if settings.rag_ocr:
            parsers.append(OcrParser())
        self._parsers = tuple(parsers)

    def load(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{file_path}")

        pages = self._parse_with_fallback(str(path))

        docs: list[Document] = []
        for page in pages:
            raw = page.text
            docs.append(
                Document(
                    text=normalize_text(raw),
                    metadata=DocumentMetadata(
                        source=str(path),
                        source_name=path.stem,
                        title=path.stem,
                        doc_type="pdf",
                        page=page.page,
                        content_type="text",
                        original_text=raw,
                        created_time=file_created_time(path),
                    ),
                )
            )

        # 表格解析：财务/经营数据表结构化入库（语义关系不丢失）
        for table in self._extract_tables(str(path)):
            docs.append(
                Document(
                    text=table.text,
                    metadata=DocumentMetadata(
                        source=str(path),
                        source_name=path.stem,
                        title=path.stem,
                        doc_type="pdf",
                        page=table.page,
                        content_type="table",
                        table_title=table.title,
                        table_headers=table.headers,
                        original_text=table.text,
                        created_time=file_created_time(path),
                    ),
                )
            )

        return docs

    def _extract_tables(self, file_path: str) -> list[TableData]:
        """提取 PDF 表格（失败不影响正文加载）。"""
        try:
            return TableParser().extract_pdf_tables(file_path)
        except Exception:  # 表格解析失败不阻断正文
            logger.warning("PDF 表格解析失败: %s", file_path, exc_info=True)
            return []

    def _parse_with_fallback(self, file_path: str) -> list[PageText]:
        """依次尝试解析器，返回首个通过质量门禁的结果；全部失败则阻断。"""
        attempts: list[str] = []
        for parser in self._parsers:
            try:
                pages = parser.parse(file_path)
            except Exception as exc:  # noqa: BLE001 缺依赖或解析异常 → 换下一个解析器
                logger.warning("PDF 解析器 %s 失败: %s", parser.name, exc)
                continue
            if not pages:
                continue
            quality = self._aggregate_quality(pages)
            if quality >= QUALITY_THRESHOLD:
                return pages
            attempts.append(f"{parser.name}={quality:.0%}")

        detail = "；".join(attempts) if attempts else "无可用解析器"
        raise PDFQualityError(
            f"PDF 文本层损坏或不可解析（质量 < {QUALITY_THRESHOLD:.0%}）：{detail}；"
            "需 OCR（Tesseract chi_sim）或干净文本版本"
        )

    @staticmethod
    def _aggregate_quality(pages: list[PageText]) -> float:
        return quality_score("\n".join(p.text for p in pages))
