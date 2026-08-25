"""OCR 解析器（Tesseract chi_sim）。

用于 PDF 文本层损坏（PyMuPDF / pdfplumber 均乱码，如小米年报）时，
渲染页面为图像 → Tesseract 中文 OCR → 恢复文本。

注意：OCR 较慢（每页 5~15s），仅在文本抽取质量不足时作为 fallback 触发。
OCR 结果仍需通过质量门禁，防止"OCR 出乱码又静默进 embedding"。
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

from app.rag.loaders.pdf.types import PageText

logger = logging.getLogger(__name__)

# 已解析的 tessdata 目录缓存
_tessdata_resolved = False

# 渲染缩放（2x 提升 OCR 精度）
_RENDER_SCALE = 2
# OCR 语言：中文简体 + 英文（财报含数字/英文）
_OCR_LANG = "chi_sim"


class OcrParser:
    """基于 PyMuPDF 渲染 + Tesseract 的 OCR 解析器。"""

    name = "ocr"

    def parse(self, file_path: str) -> list[PageText]:
        try:
            import fitz  # PyMuPDF（渲染页面为图像）
            import pytesseract
        except ImportError:
            raise ImportError(
                "OCR 需要 pymupdf + pytesseract + Tesseract 二进制（含 chi_sim 语言包），"
                "请执行：uv pip install -r requirements.txt 并安装 Tesseract"
            )

        self._ensure_environment(pytesseract)  # 定位 tesseract 可执行文件 + 语言包

        pages: list[PageText] = []
        pdf = fitz.open(file_path)
        try:
            for page in pdf:
                text = self._ocr_page(page, pytesseract)
                if not text.strip():
                    continue  # 空白页跳过
                pages.append(PageText(page=page.number + 1, text=text))
        finally:
            pdf.close()

        return pages

    @staticmethod
    def _ensure_environment(pytesseract) -> None:
        """定位 Tesseract 可执行文件与语言包，配置 pytesseract。

        conda 环境的 tesseract 不在系统 PATH，需显式指定：
            - tesseract_cmd = <env>/Library/bin/tesseract.exe
            - TESSDATA_PREFIX = <env>/share/tessdata（含 chi_sim）
        """
        global _tessdata_resolved
        if _tessdata_resolved:
            return

        tesseract = shutil.which("tesseract")
        if tesseract:
            pytesseract.pytesseract.tesseract_cmd = tesseract
        else:
            # conda 环境内常见位置
            for cand in (
                Path(sys.prefix) / "Library" / "bin" / "tesseract.exe",
                Path(sys.prefix) / "bin" / "tesseract",
            ):
                if cand.exists():
                    pytesseract.pytesseract.tesseract_cmd = str(cand)
                    tesseract = str(cand)
                    break

        if not os.environ.get("TESSDATA_PREFIX") and tesseract:
            tdir = Path(tesseract).resolve().parent
            for cand in (
                tdir / "tessdata",
                tdir.parent / "share" / "tessdata",          # <env>/Library/share/tessdata
                tdir.parent.parent / "share" / "tessdata",   # <env>/share/tessdata（conda）
            ):
                if (cand / "eng.traineddata").exists():
                    os.environ["TESSDATA_PREFIX"] = str(cand)
                    break

        _tessdata_resolved = True

    @staticmethod
    def _ocr_page(page, pytesseract) -> str:
        """渲染单页为图像并 OCR。"""
        import fitz  # 惰性导入，与 parse() 保持一致

        pix = page.get_pixmap(matrix=fitz.Matrix(_RENDER_SCALE, _RENDER_SCALE))
        img = pix.pil_image()
        try:
            return pytesseract.image_to_string(img, lang=_OCR_LANG)
        except Exception as exc:  # noqa: BLE001 单页 OCR 失败不阻断整份
            logger.warning("OCR 第 %s 页失败: %s", page.number + 1, exc)
            return ""
