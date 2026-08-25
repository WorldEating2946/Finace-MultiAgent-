"""PDF 完整入库 + 检索可追溯性测试（Dummy embedding 保持快速确定性）。"""

from app.rag.document import DocumentChunk
from app.rag.embedding import DummyEmbeddingModel
from app.rag.ingestion import ingest
from app.rag.pipeline import retrieve
from app.rag.vector_store import FAISSVectorStore

DIM = 128


def _make_model() -> DummyEmbeddingModel:
    return DummyEmbeddingModel(dim=DIM)


def _make_store() -> FAISSVectorStore:
    return FAISSVectorStore(dim=DIM)


def test_pdf_full_ingestion(xiaomi_pdf_path):
    """PDF 应完整走 入库链路：每页 → Document → 切分 → 写入向量库。"""
    model = _make_model()
    store = _make_store()

    chunks = ingest(str(xiaomi_pdf_path), company="小米", _model=model, _store=store)

    assert len(chunks) > 0
    for c in chunks[:10]:
        assert isinstance(c, DocumentChunk)
        assert c.company == "小米"
        assert c.doc_type == "pdf"
        assert c.page is not None and c.page >= 1
        assert c.metadata["source"]
        assert c.metadata["doc_type"] == "pdf"


def test_pdf_retrieve_is_traceable(xiaomi_pdf_path):
    """检索结果应可追溯：带页码与来源（企业问答必需）。"""
    model = _make_model()
    store = _make_store()
    ingest(str(xiaomi_pdf_path), company="小米", _model=model, _store=store)

    result = retrieve(
        "小米主要业务有哪些", company="小米", top_k=3, _model=model, _store=store
    )

    assert len(result.chunks) >= 1
    chunk = result.chunks[0]
    assert chunk.page is not None            # 可追溯：页码
    assert chunk.metadata["source"]          # 可追溯：来源文件
    assert chunk.metadata["doc_type"] == "pdf"
    assert chunk.metadata["section"]         # 可追溯：章节路径（Structure-aware）


def test_pdf_pages_are_distinct_documents(xiaomi_pdf_path):
    """不同页应生成不同 Document（page 递增）。"""
    from app.rag.loaders.pdf_loader import PDFLoader

    docs = PDFLoader().load(str(xiaomi_pdf_path))

    pages = [d.metadata.page for d in docs]
    assert pages == sorted(set(pages))  # 页码唯一且递增
    assert pages[0] == 1
