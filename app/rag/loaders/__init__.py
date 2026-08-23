"""文档加载器集合 + 文件类型分发。

统一入口：
    from app.rag import load_documents     # 检测文件类型 → 对应 Loader → list[Document]
    from app.rag.loaders import get_loader  # 按扩展名取 Loader 实例

当前支持：.md / .markdown / .txt / .pdf；Word（含 OCR、表格、复杂布局）后续支持。
"""

from pathlib import Path

from app.rag.document import Document
from app.rag.loaders.base import DocumentLoader
from app.rag.loaders.markdown_loader import MarkdownLoader
from app.rag.loaders.pdf_loader import PDFLoader
from app.rag.loaders.txt_loader import TXTLoader

__all__ = [
    "Document",
    "DocumentLoader",
    "MarkdownLoader",
    "PDFLoader",
    "TXTLoader",
    "get_loader",
    "load_documents",
]

# 扩展名 → Loader 实例（无状态，可复用）
_LOADERS: dict[str, DocumentLoader] = {
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".txt": TXTLoader(),
    ".pdf": PDFLoader(),
}

_SUPPORTED_EXTENSIONS = " / ".join(sorted(_LOADERS))


def get_loader(file_path: str) -> DocumentLoader:
    """按文件扩展名返回对应 Loader。

    Raises:
        ValueError: 不支持的文件类型。
    """
    ext = Path(file_path).suffix.lower()
    try:
        return _LOADERS[ext]
    except KeyError:
        raise ValueError(
            f"暂不支持的文件类型：{ext}\n"
            f"当前支持：{_SUPPORTED_EXTENSIONS}\n"
            f"Word（含 OCR、表格、复杂布局）计划后续支持"
        )


def load_documents(file_path: str) -> list[Document]:
    """统一加载入口：检测文件类型 → 对应 Loader → list[Document]。

    Args:
        file_path: 文件路径。

    Returns:
        list[Document]: 原始文档列表（未切分，切分交给 splitter）。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 路径不是文件，或不支持的文件类型。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")
    if not path.is_file():
        raise ValueError(f"路径不是文件：{file_path}")

    return get_loader(file_path).load(file_path)
