"""Markdown 文档加载器。

metadata（DocumentMetadata 统一规范）：
    source / source_name / doc_type="markdown" / page=None / created_time

Phase 1 优先实现 Markdown：企业技术文档大量 Markdown，
且可用来验证 splitter 的标题感知切分策略。
"""

from pathlib import Path

from app.rag.document import Document, DocumentMetadata
from app.rag.loaders.base import DocumentLoader, file_created_time
from app.rag.parsers.text_normalizer import normalize_text


class MarkdownLoader(DocumentLoader):
    """加载 .md / .markdown 文件为 Document。"""

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
                    doc_type="markdown",
                    page=None,
                    original_text=raw,
                    created_time=file_created_time(path),
                ),
            )
        ]
