"""离线知识入库单元测试。

测试 ingest() 完整链路：
file_path → load → split → embed → vector_store.add()
"""

from app.rag.document import DocumentChunk
from app.rag.embedding import DummyEmbeddingModel
from app.rag.ingestion import ingest
from app.rag.pipeline import retrieve
from app.rag.vector_store import FAISSVectorStore

DIM = 128


# ── 测试辅助 ──────────────────────────────────────────────────────

def _make_model(dim: int = DIM) -> DummyEmbeddingModel:
    return DummyEmbeddingModel(dim=dim)


def _make_store(dim: int = DIM) -> FAISSVectorStore:
    return FAISSVectorStore(dim=dim)


# ── 核心功能 ──────────────────────────────────────────────────────

def test_ingest_returns_chunks(tmp_path):
    """入库应返回切分后的 DocumentChunk 列表。"""
    file_path = tmp_path / "测试文档.md"
    file_path.write_text("# 基金赎回规则\n\n基金赎回通常需要T+1到T+3个工作日到账。", encoding="utf-8")

    model = _make_model()
    store = _make_store()
    chunks = ingest(str(file_path), company="测试公司", _model=model, _store=store)

    assert len(chunks) >= 1
    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert c.company == "测试公司"


def test_ingest_fills_company_on_all_chunks(tmp_path):
    """入库后所有 chunk 的 company 字段应与 ingest 参数一致。"""
    file_path = tmp_path / "长文档.txt"
    # 写入足够长的文本以确保 splitter 产生多个 chunk
    file_path.write_text("基金投资知识。\n\n" * 200, encoding="utf-8")

    model = _make_model()
    store = _make_store()
    chunks = ingest(str(file_path), company="宁德时代", _model=model, _store=store)

    assert len(chunks) > 1, f"预期多 chunk，实际 {len(chunks)}"
    for c in chunks:
        assert c.company == "宁德时代", f"chunk {c.chunk_id} company={c.company}"


def test_ingest_then_retrieve_e2e(tmp_path):
    """ingest → retrieve 端到端：入库后可检索到相关内容。"""
    # 1. 准备文档
    file_path = tmp_path / "基金规则.md"
    file_path.write_text(
        "# 基金赎回规则\n\n"
        "基金赎回通常需要T+1到T+3个工作日到账。\n\n"
        "货币基金赎回最快T+0到账。\n\n"
        "# 基金申购规则\n\n"
        "基金申购确认需要T+1个工作日。",
        encoding="utf-8",
    )

    # 2. 入库
    model = _make_model()
    store = _make_store()
    chunks = ingest(str(file_path), company="测试公司", _model=model, _store=store)
    assert len(chunks) >= 1

    # 3. 检索
    result = retrieve("基金赎回多久到账", company="测试公司", _model=model, _store=store)

    assert result.query == "基金赎回多久到账"
    assert len(result.chunks) > 0
    assert result.confidence > 0.0


def test_ingest_source_metadata_preserved(tmp_path):
    """入库后 chunk 应保留 source / source_name / chunk_id 等溯源字段。"""
    file_path = tmp_path / "金融政策.md"
    file_path.write_text("# 金融政策\n\n这是关于金融监管政策的文档内容。", encoding="utf-8")

    model = _make_model()
    store = _make_store()
    chunks = ingest(str(file_path), company="测试", _model=model, _store=store)

    for c in chunks:
        assert c.source, "source 不应为空"
        assert c.source_name, "source_name 不应为空"
        assert c.chunk_id, "chunk_id 不应为空"
        assert c.text, "text 不应为空"


# ── 公共 API 验证 ─────────────────────────────────────────────────

def test_ingest_via_public_api(tmp_path):
    """from app.rag import ingest 应可用。"""
    from app.rag import ingest as public_ingest

    file_path = tmp_path / "文档.txt"
    file_path.write_text("测试内容。", encoding="utf-8")

    model = _make_model()
    store = _make_store()
    chunks = public_ingest(str(file_path), company="测试", _model=model, _store=store)

    assert len(chunks) >= 1


# ── 错误处理 ──────────────────────────────────────────────────────

def test_ingest_missing_file_raises():
    """不存在的文件应抛出 FileNotFoundError。"""
    import pytest

    model = _make_model()
    store = _make_store()

    with pytest.raises(FileNotFoundError):
        ingest("/不存在的文件.md", company="测试", _model=model, _store=store)


def test_ingest_unsupported_extension_raises(tmp_path):
    """不支持的文件类型应抛出 ValueError。"""
    import pytest

    file_path = tmp_path / "文档.docx"
    file_path.write_text("fake docx", encoding="utf-8")

    model = _make_model()
    store = _make_store()

    with pytest.raises(ValueError):
        ingest(str(file_path), company="测试", _model=model, _store=store)


# ── 多公司知识库隔离（配置化路径 + 独立子目录）────────────────────

def test_multi_company_knowledge_isolation(monkeypatch, tmp_path):
    """不同 company 应持久化到独立子目录，且检索互相隔离。

    使用 DummyEmbeddingModel 避免加载真实 BGE-M3，聚焦隔离逻辑。
    """
    import app.rag.vector_store as vs
    from app.rag.pipeline import retrieve

    dummy = DummyEmbeddingModel(dim=DIM)
    monkeypatch.setattr(vs.settings, "rag_vector_store_path", str(tmp_path))
    monkeypatch.setattr(vs, "_default_stores", {})
    monkeypatch.setattr("app.rag.ingestion.get_embedding_model", lambda: dummy)
    monkeypatch.setattr("app.rag.retriever.get_embedding_model", lambda: dummy)

    file_a = tmp_path / "公司A.md"
    file_a.write_text("# 公司A\n\n宁德时代动力电池龙头，全球装机量第一。", encoding="utf-8")
    file_b = tmp_path / "公司B.md"
    file_b.write_text("# 公司B\n\n比亚迪新能源汽车，出口欧洲市场。", encoding="utf-8")

    ingest(str(file_a), company="company_a")
    ingest(str(file_b), company="company_b")

    # 各公司独立存档目录
    assert (tmp_path / "company_a" / "index.faiss").exists()
    assert (tmp_path / "company_b" / "index.faiss").exists()

    # 检索 company_a 只返回 company_a 的 chunk（知识库互相隔离）
    result = retrieve("动力电池", company="company_a")
    assert len(result.chunks) >= 1
    assert all(c.company == "company_a" for c in result.chunks)
