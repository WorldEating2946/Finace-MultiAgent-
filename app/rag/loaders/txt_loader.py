"""纯文本（TXT）文档加载器。

metadata（DocumentMetadata 统一规范）：
    source / source_name / doc_type="text" / page=None / created_time
"""

from pathlib import Path

from app.rag.document import Document, DocumentMetadata
from app.rag.loaders.base import DocumentLoader, file_created_time
from app.rag.parsers.text_normalizer import normalize_text


class TXTLoader(DocumentLoader):
    """加载 .txt 文件为 Document。"""

    def load(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{file_path}")

        raw = path.read_text(encoding="utf-8")
        return [
            Document(
                text=normalize_text(raw),
                metadata=DocumentMetadata(
                    source=str(path),
                    source_name=path.stem,
                    title=path.stem,
                    doc_type="text",
                    page=None,
                    original_text=raw,
                    created_time=file_created_time(path),
                ),
            )
        ]
