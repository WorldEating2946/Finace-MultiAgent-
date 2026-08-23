"""PR44.4 向量后端健康检查单测（离线，无需真实 Milvus / pymilvus）。

策略：
    - milvus 检查注入 ContractFakeMilvusClient（已扩展 get_server_version /
      list_databases / describe_collection，对齐真实 env v2.4.0）；
    - faiss 检查 monkeypatch settings.rag_vector_store_path；
    - 不触碰共享真实 Milvus（只读约束下单测零网络）。
"""

from app.core.config import settings
from app.rag.vectorstore import check_backend_ready
from app.rag.vectorstore.factory import _resolve_backend
from tests.milvus_fake_client import ContractFakeMilvusClient

_COLLECTION = "finance_knowledge"


def _ready_fake(dim: int = 1024) -> ContractFakeMilvusClient:
    """就绪状态的 fake：finance_agent 库 + finance_knowledge collection（dim 匹配）。"""
    fake = ContractFakeMilvusClient()
    fake.create_collection(_COLLECTION, dimension=dim)
    return fake


# ── FAISS ───────────────────────────────────────────────────────

def test_check_faiss_ready(monkeypatch, tmp_path):
    """数据目录存在 → 空问题列表。"""
    monkeypatch.setattr(settings, "rag_vector_store_path", str(tmp_path))
    assert check_backend_ready(backend="faiss") == []


def test_check_faiss_missing_dir(monkeypatch, tmp_path):
    """数据目录缺失 → 明确报错（部署前该目录由 ingest/init 建立）。"""
    missing = tmp_path / "no_such_dir"
    monkeypatch.setattr(settings, "rag_vector_store_path", str(missing))
    problems = check_backend_ready(backend="faiss")
    assert len(problems) == 1
    assert "不存在" in problems[0]


# ── Milvus 顺序检查 ────────────────────────────────────────────

def test_check_milvus_ready():
    """可达 + 库存在 + collection 存在 + 维度匹配 → 就绪。"""
    assert check_backend_ready(backend="milvus", client=_ready_fake()) == []


def test_check_milvus_unreachable():
    """服务不可达 → 第一项即 fail fast，返回可达性错误。"""

    class _Unreachable:
        def get_server_version(self) -> str:
            raise RuntimeError("connection refused")

    problems = check_backend_ready(backend="milvus", client=_Unreachable())
    assert len(problems) == 1
    assert "不可达" in problems[0]


def test_check_milvus_missing_database():
    """业务库不存在 → 明确报错（MilvusClient 对缺库不抛错，必须查 list_databases）。"""
    fake = _ready_fake()
    fake._databases = ["default"]  # 模拟共享环境只有 default 库
    problems = check_backend_ready(backend="milvus", client=fake)
    assert len(problems) == 1
    assert "database" in problems[0] and "不存在" in problems[0]


def test_check_milvus_missing_collection():
    """collection 不存在 → 指向迁移脚本。"""
    fake = ContractFakeMilvusClient()  # 库在、collection 无
    problems = check_backend_ready(backend="milvus", client=fake)
    assert len(problems) == 1
    assert "collection" in problems[0] and "migrate" in problems[0]


def test_check_milvus_dimension_mismatch():
    """collection 维度 ≠ 配置 → 拒绝启动（数据源不一致）。"""
    fake = _ready_fake(dim=128)  # BGE-M3 应为 1024
    problems = check_backend_ready(backend="milvus", client=fake)
    assert len(problems) == 1
    assert "维度不匹配" in problems[0]


def test_check_milvus_no_embedding_field():
    """collection 缺 embedding 字段 → schema 与 AD-2 不符。"""

    class _NoEmbedding:
        def get_server_version(self) -> str:
            return "v2.4.0"

        def list_databases(self) -> list[str]:
            return ["default", "finance_agent"]

        def has_collection(self, name: str) -> bool:
            return True

        def describe_collection(self, name: str) -> dict:
            return {"fields": []}

    problems = check_backend_ready(backend="milvus", client=_NoEmbedding())
    assert len(problems) == 1
    assert "embedding" in problems[0]


# ── 边界 ───────────────────────────────────────────────────────

def test_unknown_backend():
    """未知后端 → 明确报错。"""
    problems = check_backend_ready(backend="pgvector")
    assert len(problems) == 1
    assert "未知" in problems[0]


def test_resolve_backend_defaults_to_settings():
    """工厂 backend 解析：显式优先，否则读 settings（PR44.4 配置化默认）。"""
    original = settings.rag_vector_backend
    try:
        settings.rag_vector_backend = "milvus"
        assert _resolve_backend(None) == "milvus"  # 未传 → 配置
        assert _resolve_backend("") == "milvus"  # 空串 → 配置
        assert _resolve_backend("faiss") == "faiss"  # 显式优先
    finally:
        settings.rag_vector_backend = original
