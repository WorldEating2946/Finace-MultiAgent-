"""PR44.3.1 Milvus 实现（VectorStore ABC）。

与 FAISSStore 的本质差异（docs/pr44_milvus_design.md AD-2/AD-4/AD-7）：
    - 持久化：Milvus 服务端管理，无 save()/load()/compact()（不继承 LocalVectorStoreMixin）；
    - 过滤：structured filters 翻译为 Milvus expr，原生标量过滤（非 Python 后置）；
    - 删除：物理删除（delete by expr），非逻辑删除；
    - 幂等 add：upsert（主键 chunk_id 覆盖），非手动去重；
    - 维度：collection 创建时固定（settings.milvus_dim，BGE-M3=1024），非运行时推断。

pymilvus 为可选依赖（懒加载）——本模块 import 不触发 pymilvus；
未安装时首次操作抛 ImportError（含安装指引）。
"""

from __future__ import annotations

import hashlib
import re

from app.rag.vectorstore.base import VectorStore
from app.rag.vectorstore.models import SearchResult, VectorRecord

# AD-2 冻结 schema：finance_knowledge 标量字段（(name, type)）。
# embedding（FLOAT_VECTOR, dim）与 metadata（JSON）单独处理。
_SCHEMA_FIELDS: list[tuple[str, str]] = [
    ("chunk_id", "VARCHAR"),
    ("company_id", "VARCHAR"),
    ("document_id", "VARCHAR"),
    ("year", "INT64"),
    ("section", "VARCHAR"),
    ("text", "VARCHAR"),
]
# 空 company_id 时兜底的"恒真"表达式：Milvus query() 空 filter 必须配 limit，
# 用非空恒真表达式规避 "empty expression should be used with limit"。
_ALL_EXPR = 'chunk_id != ""'

# VARCHAR 字段 max_length（pymilvus 建 schema 必填）。
_SCHEMA_MAX_LENGTH: dict[str, int] = {
    "chunk_id": 128,
    "company_id": 64,
    "document_id": 128,
    "section": 256,
    "text": 8192,
}


class MilvusStore(VectorStore):
    """Milvus 向量库（只实现 VectorStore ABC 的 add/search/delete/update/count）。

    Args:
        dim:             向量维度（BGE-M3=1024；collection 创建时固定）。
        uri:             Milvus 服务地址（http://... 或 "./milvus.db" Lite）。
        collection_name: Collection 名（AD-2 finance_knowledge；所有公司共享，
                         通过 company_id 字段做多公司隔离）。
        company_id:      所属企业（一级过滤上下文；search/delete 缺省过滤用）。
        db_name:         Milvus Database（AD-1 冻结 = finance_agent；默认即业务库，
                         禁止落到 default 库——本实例与 edu_agent 共享）。
        client:          注入的 client（测试传 ContractFakeMilvusClient）；
                         为 None 时懒加载 pymilvus 创建真实 client。
    """

    def __init__(
        self,
        dim: int = 1024,
        uri: str = "http://localhost:19530",
        collection_name: str = "finance_knowledge",
        company_id: str = "",
        db_name: str = "finance_agent",  # AD-1 业务库；勿传 ""（= default 库）
        client: object | None = None,
    ):
        self._dim = dim
        self._uri = uri
        self._db_name = db_name
        self._collection = collection_name
        self._company_id = company_id
        self._injected_client = client  # 测试注入 fake；None = 懒加载
        self._real_client: object | None = None  # 懒加载的真实 client 缓存

    @property
    def _client(self) -> object:
        """返回注入 client，或懒加载真实 pymilvus MilvusClient。"""
        if self._injected_client is not None:
            return self._injected_client
        if self._real_client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:  # pragma: no cover - 无 pymilvus 的报错路径
                raise ImportError(
                    "MilvusStore 需要 pymilvus，请先安装：pip install pymilvus"
                ) from exc
            self._real_client = MilvusClient(uri=self._uri, db_name=self._db_name)
        return self._real_client

    # ── VectorStore ABC ─────────────────────────────────────────

    def add(self, records: list[VectorRecord]) -> None:
        """批量写入（upsert，主键 chunk_id 幂等覆盖）。"""
        if not records:
            return
        self._ensure_collection()
        self._client.upsert(  # type: ignore[attr-defined]
            collection_name=self._collection,
            data=[self._to_row(r) for r in records],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Dense 检索 + filters → Milvus expr 原生标量过滤（非 Python 后置）。"""
        client = self._client
        if not client.has_collection(self._collection):  # type: ignore[attr-defined]
            return []
        expr = self._build_expr(self._merge_company_filter(filters))
        hits = client.search(  # type: ignore[attr-defined]
            collection_name=self._collection,
            data=[query_embedding],
            anns_field="embedding",
            filter=expr or "",
            output_fields=["chunk_id", "text", "metadata"],
            limit=top_k,
        )
        return [self._to_result(h) for h in hits[0]] if hits else []

    def delete(self, ids: list[str]) -> int:
        """物理删除（expr: chunk_id in [...]），返回实际删除数。"""
        ids = [i for i in ids if i]
        if not ids:
            return 0
        client = self._client
        if not client.has_collection(self._collection):  # type: ignore[attr-defined]
            return 0
        quoted = ", ".join(f'"{i}"' for i in ids)
        expr = f"chunk_id in [{quoted}]"
        if self._company_id:  # 多公司隔离：同一 chunk_id 不会误删其他公司
            expr += f' and company_id == "{self._company_id}"'
        result = client.delete(  # type: ignore[attr-defined]
            collection_name=self._collection, filter=expr
        )
        return int(result.get("delete_count", 0))

    def update(self, record: VectorRecord) -> bool:
        """更新 = upsert（同主键覆盖）。返回 True 表示原记录存在。"""
        existed = False
        client = self._client
        if client.has_collection(self._collection):  # type: ignore[attr-defined]
            rows = client.query(  # type: ignore[attr-defined]
                collection_name=self._collection,
                filter=f'chunk_id == "{record.chunk_id}"',
                output_fields=["chunk_id"],
            )
            existed = bool(rows)
        self.add([record])
        return existed

    def count(self) -> int:
        """返回本公司活跃记录数（query count(*)；stats 的 row_count 含已删行不可用）。

        共享 collection 多公司共存 → 用构造时 company_id 过滤，与 FAISSStore
        （per-company 实例）的 count 语义一致。
        """
        client = self._client
        if not client.has_collection(self._collection):  # type: ignore[attr-defined]
            return 0
        expr = (
            f'company_id == "{self._company_id}"' if self._company_id else _ALL_EXPR
        )
        rows = client.query(  # type: ignore[attr-defined]
            collection_name=self._collection, filter=expr, output_fields=["count(*)"]
        )
        return int(rows[0].get("count(*)", 0)) if rows else 0

    # ── Hybrid 支持（AD-5 方法表：喂 BM25 语料 + 结果回填 dense_vector）──

    def all_chunks(self) -> list[VectorRecord]:
        """返回本公司全部活跃 chunk（供 BM25 稀疏检索构建语料）。

        共享 collection 多公司共存 → 用构造时 company_id 过滤，语义与
        FAISSStore（per-company 实例）一致。
        """
        client = self._client
        if not client.has_collection(self._collection):  # type: ignore[attr-defined]
            return []
        expr = (
            f'company_id == "{self._company_id}"' if self._company_id else _ALL_EXPR
        )
        rows = client.query(  # type: ignore[attr-defined]
            collection_name=self._collection,
            filter=expr,
            output_fields=["chunk_id", "embedding", "metadata", "text"],
        )
        return [
            VectorRecord(
                chunk_id=r.get("chunk_id", ""),
                text=r.get("text", ""),
                embedding=r.get("embedding") or [],
                metadata=dict(r.get("metadata") or {}),
            )
            for r in rows
        ]

    def vector_of(self, chunk_id: str) -> list[float] | None:
        """返回指定 chunk 的稠密向量（供结果回填 dense_vector）。"""
        client = self._client
        if not client.has_collection(self._collection):  # type: ignore[attr-defined]
            return None
        rows = client.query(  # type: ignore[attr-defined]
            collection_name=self._collection,
            filter=f'chunk_id == "{chunk_id}"',
            output_fields=["embedding"],
        )
        return rows[0].get("embedding") if rows else None

    # ── 内部辅助 ─────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """collection 不存在时创建（AD-2 schema；fake 用精简签名）。"""
        client = self._client
        if client.has_collection(self._collection):  # type: ignore[attr-defined]
            return
        if getattr(client, "_CONTRACT_FAKE", False):
            # 测试 fake：接收精简 schema 说明（真实 pymilvus 走下面分支）
            client.create_collection(  # type: ignore[attr-defined]
                collection_name=self._collection,
                dimension=self._dim,
                metric_type="COSINE",
                schema=list(_SCHEMA_FIELDS),
                primary_field="chunk_id",
            )
            return
        self._create_collection_pymilvus()

    def _create_collection_pymilvus(self) -> None:
        """真实 pymilvus：schema 声明标量字段 + metadata JSON + FLAT/COSINE 索引。"""
        from pymilvus import DataType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        for name, typ in _SCHEMA_FIELDS:
            schema.add_field(
                name,
                getattr(DataType, typ),
                is_primary=(name == "chunk_id"),
                max_length=_SCHEMA_MAX_LENGTH.get(name),
            )
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)
        index_params = MilvusClient.prepare_index_params()
        # AD-4：开发用 FLAT 作正确性基准；生产 Benchmark（PR44.3.3）后再换 HNSW/IVF_FLAT
        index_params.add_index(
            field_name="embedding", index_type="FLAT", metric_type="COSINE"
        )
        self._client.create_collection(  # type: ignore[attr-defined]
            self._collection, schema=schema, index_params=index_params
        )

    def _merge_company_filter(self, filters: dict | None) -> dict:
        """filters 缺 company_id 时用构造时 company_id 兜底（与 FAISSStore 一致）。"""
        merged = dict(filters or {})
        if "company_id" not in merged and self._company_id:
            merged["company_id"] = self._company_id
        return merged

    @staticmethod
    def _build_expr(filters: dict | None) -> str | None:
        """结构化 filters → Milvus expr（AD-5 过滤语义，与 FAISS 对齐）。"""
        if not filters:
            return None
        parts: list[str] = []
        for key, value in filters.items():
            if value is None:
                continue
            if key == "company_id":
                parts.append(f'company_id == "{value}"')
            elif key == "year":
                parts.append(f"year >= {int(value)}")
            elif key == "document_type":
                # 别名映射：外部 document_type → metadata JSON 的 source_type
                parts.append(f'metadata["source_type"] == "{value}"')
            elif key == "section":
                parts.append(f'section == "{value}"')
            elif isinstance(value, str):
                parts.append(f'metadata["{key}"] == "{value}"')
            else:
                parts.append(f'metadata["{key}"] == "{value}"')
        return " and ".join(parts) if parts else None

    def _to_row(self, record: VectorRecord) -> dict:
        """VectorRecord → Milvus row dict（标量字段顶层 + 全量 metadata JSON）。

        标量字段（company_id / document_id / year / section）在 schema 里非空且无默认，
        故此处**始终**补齐四列的兜底值——否则直接入库(ingest)路径缺失 document_id/year
        会触发 Milvus "Insert missed an field ..." 异常。兜底逻辑与 PR44.3.2 迁移补全一致：
            document_id ← meta.document_id ?? md5(source)
            year        ← meta.year ?? 从 source(source_name) 提取 20xx
            company_id  ← meta.company_id ?? self._company_id ?? meta.company
            section     ← meta.section ?? ""
        """
        meta = dict(record.metadata)
        source = meta.get("source") or meta.get("source_name") or record.chunk_id

        doc_id = meta.get("document_id") or hashlib.md5(str(source).encode("utf-8")).hexdigest()

        year = meta.get("year")
        if year is None:
            m = re.search(r"(20\d{2})", str(source))
            year = int(m.group(1)) if m else 0

        company = meta.get("company_id") or self._company_id or meta.get("company") or ""
        section = meta.get("section") or ""

        # 保持 metadata JSON 与顶层一致（供 _to_result 回读 / search 过滤）
        meta.setdefault("company_id", company)
        meta.setdefault("document_id", doc_id)
        meta.setdefault("year", year)
        meta.setdefault("section", section)

        return {
            "chunk_id": record.chunk_id,
            "text": record.text,
            "embedding": record.embedding,
            "metadata": meta,
            "company_id": company,
            "document_id": doc_id,
            "year": year,
            "section": section,
        }

    @staticmethod
    def _to_result(hit: dict) -> SearchResult:
        """Milvus search hit → SearchResult（score 收敛到 [0,1] 契约）。"""
        entity = hit.get("entity") or {}
        metadata = dict(entity.get("metadata") or {})
        score = float(hit.get("distance", 0.0))
        return SearchResult(
            chunk_id=hit.get("id", "") or entity.get("chunk_id", ""),
            text=entity.get("text", ""),
            score=max(0.0, min(1.0, score)),
            metadata=metadata,
        )
