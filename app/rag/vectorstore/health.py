"""PR44.4 向量后端启动健康检查（fail fast，无自动 fallback）。

生产切 milvus 前先确认：服务可达 / 业务库存在 / collection 存在 / 维度匹配。
**全部只读**——尊重共享 Milvus 的"只能新建、不动他库"约束，检查绝不创建/修改任何对象
（创建仅由 scripts/migrate_faiss_to_milvus.py 负责）。

FAISS（默认/离线评测/紧急回滚）只做轻量检查：数据目录存在。

调用：FastAPI lifespan 启动时 `check_backend_ready()`；有错 raise → 应用拒绝启动。
v1 策略 = fail fast，不做 milvus→faiss 自动 fallback（数据一致性未实时保证，
自动降级会静默产生结果差异）。
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings

_KNOWN_BACKENDS = ("faiss", "milvus")


def check_backend_ready(
    backend: str | None = None,
    *,
    client: object | None = None,
) -> list[str]:
    """返回后端就绪检查的问题列表；空列表 = 就绪。

    Args:
        backend: None → 读 settings.rag_vector_backend；显式 "faiss" / "milvus"。
        client:  测试 seam —— 注入 in-memory fake MilvusClient；
                 为 None 时懒加载 pymilvus 创建真实 client（仅 backend=milvus 时）。

    Returns:
        list[str]: 每个已发现的问题；空 = 检查通过。

    Raises:
        ValueError: backend 未知（Literal 配置已在 settings 层拦截，此处兜底）。
    """
    backend = (backend or settings.rag_vector_backend).lower()
    if backend == "faiss":
        return _check_faiss()
    if backend == "milvus":
        return _check_milvus(client=client)
    return [f"未知向量后端: {backend!r}（支持 {'/'.join(_KNOWN_BACKENDS)}）"]


def _check_faiss() -> list[str]:
    """FAISS：数据根目录可读（生产归档默认在 settings.rag_vector_store_path）。"""
    path = Path(settings.rag_vector_store_path)
    if not path.exists():
        return [f"FAISS 数据目录不存在: {path}"]
    return []


def _check_milvus(*, client: object | None = None) -> list[str]:
    """Milvus：可达 → 业务库存在 → collection 存在 → 维度匹配。

    顺序 fail-fast：更基础的问题（不可达/库缺失）先行返回，避免错误级联。
    """
    if client is None:
        try:
            from pymilvus import MilvusClient
        except ImportError:
            return ["pymilvus 未安装，请先：pip install pymilvus"]
        try:
            client = MilvusClient(
                uri=settings.milvus_uri, db_name=settings.milvus_db_name
            )
        except Exception as exc:  # noqa: BLE001  # 一次性启动检查：任何连接异常都转可读报错
            return [f"Milvus 连接失败 {settings.milvus_uri}: {exc}"]
    # 1. 服务可达
    try:
        client.get_server_version()  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        return [f"Milvus 不可达 {settings.milvus_uri}: {exc}"]
    # 2. 业务库存在——MilvusClient 对不存在的 db 不报错（has_collection 返回 False），
    #    必须显式查 list_databases 才能给出准确原因。
    try:
        databases = list(client.list_databases())  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        return [f"Milvus list_databases 失败: {exc}"]
    if settings.milvus_db_name not in databases:
        return [
            (
                f"Milvus database '{settings.milvus_db_name}' 不存在"
                "（需先跑 scripts/migrate_faiss_to_milvus.py）"
            )
        ]
    # 3. collection 存在（不存在的 collection 返回 False 不抛）
    try:
        exists = bool(
            client.has_collection(settings.milvus_collection_name)  # type: ignore[attr-defined]
        )
    except Exception as exc:  # noqa: BLE001
        return [f"Milvus has_collection 失败: {exc}"]
    if not exists:
        return [
            (
                f"Milvus collection '{settings.milvus_collection_name}' 不存在"
                "（需先跑 scripts/migrate_faiss_to_milvus.py 迁移）"
            )
        ]
    # 4. 维度匹配——collection 创建时固定（AD-2），不匹配说明数据源与配置不一致
    try:
        desc = client.describe_collection(  # type: ignore[attr-defined]
            settings.milvus_collection_name
        )
        dim = _embedding_dim(desc)
    except Exception as exc:  # noqa: BLE001
        return [f"Milvus describe_collection 失败: {exc}"]
    if dim is None:
        return ["Milvus collection 缺少 embedding 字段（schema 与 AD-2 不符）"]
    if dim != settings.milvus_dim:
        return [
            (
                f"向量维度不匹配: Milvus={dim} 配置 milvus_dim={settings.milvus_dim}"
                "（数据源与配置不一致，拒绝启动）"
            )
        ]
    return []


def _embedding_dim(desc: dict) -> int | None:
    """从 describe_collection 结果提取 embedding 字段维度（pymilvus 3.x 结构）。"""
    for field in desc.get("fields", []):
        if field.get("name") == "embedding":
            return (field.get("params") or {}).get("dim")
    return None
