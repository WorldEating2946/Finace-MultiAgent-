"""Loader 统一接口（文件 → Document）。

所有具体 Loader 遵循同一契约：
    输入文件 → list[Document]（原始文档，未切分）

后续扩展（PDF / Word / 网页抓取等）新增具体 Loader 即可，
主流程（splitter / ingestion）零改动。
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from app.rag.document import Document


def file_created_time(path: Path) -> str:
    """取文件修改时间作为 ISO 时间字符串（DocumentMetadata.created_time）。

    Args:
        path: 源文件路径。

    Returns:
        ISO 格式时间字符串（UTC）；读取失败时返回空串。
    """
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


class DocumentLoader(ABC):
    """统一文档加载接口。"""

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        """加载文件为 Document 列表（一个文件通常对应一个 Document）。

        Args:
            file_path: 源文件路径。

        Returns:
            list[Document]: 原始文档列表。

        Raises:
            FileNotFoundError: 文件不存在。
        """
        raise NotImplementedError
