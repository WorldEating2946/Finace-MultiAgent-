"""PR44.3.2 迁移工具单元测试（离线，无需 Milvus 服务）。

策略（对齐 test_milvus_adapter.py）：
  - 纯函数：derive_document_id / derive_year / enrich_metadata / batch_iter；
  - 集成：migrate_company + validate_migration + rollback_company 用 真实 FAISSStore
    （内存 dummy）+ 注入 ContractFakeMilvusClient 的 MilvusStore——零 pymilvus / 零 Milvus 依赖；
  - 校验必须能检测到：计数不一致 / embedding 篡改 / metadata 篡改。

与 PR44.3.1 前置核查的对应：
  - 元数据补全：document_id=md5(source)、year=source_name 正则（Q2 结论）；
  - embedding 复用不重算：cosine >= 0.9999（Q3 结论）。
"""

from __future__ import annotations

import sys
from hashlib import md5
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.embedding import DummyEmbeddingModel
from app.rag.vectorstore import FAISSStore, MilvusStore, VectorRecord
from scripts.migrate_faiss_to_milvus import (
    _milvus_rows,
    batch_iter,
    derive_document_id,
    derive_year,
    enrich_metadata,
    enrich_record,
    migrate_company,
    rollback_company,
    validate_migration,
)
from tests.milvus_fake_client import ContractFakeMilvusClient

DIM = 128


# ── 测试辅助 ──────────────────────────────────────────────────────

def _make_record(
    text: str,
    chunk_id: str,
    *,
    company_id: str = "xiaomi",
    source: str = "小米2025年报.pdf",
    source_name: str = "小米2025年报",
    **extra,
) -> VectorRecord:
    """创建带 source/source_name 的 VectorRecord（供 document_id/year 派生）。"""
    meta = {
        "company_id": company_id,
        "source": source,
        "source_name": source_name,
        "source_type": "annual_report",
    }
    meta.update(extra)
    return VectorRecord(
        chunk_id=chunk_id,
        text=text,
        embedding=DummyEmbeddingModel(dim=DIM).embed([text])[0],
        metadata=meta,
    )


def _fake_milvus() -> MilvusStore:
    return MilvusStore(
        dim=DIM,
        uri="fake://unused",
        collection_name="test_finance_knowledge",
        company_id="xiaomi",
        client=ContractFakeMilvusClient(),
    )


def _faiss_store() -> FAISSStore:
    return FAISSStore(dim=DIM, company="xiaomi")


# ── A. 纯函数：元数据补全 ────────────────────────────────────────

def test_derive_document_id_from_source():
    """document_id = md5(source)。"""
    did = derive_document_id({"source": "小米2025年报.pdf"})
    assert did == md5("小米2025年报.pdf".encode()).hexdigest()


def test_derive_document_id_fallback_source_name():
    """source 缺失时退 source_name。"""
    did = derive_document_id({"source_name": "小米2025年报"})
    assert did == md5("小米2025年报".encode()).hexdigest()


def test_derive_document_id_none_when_missing():
    """source / source_name 都缺 → None。"""
    assert derive_document_id({"company_id": "xiaomi"}) is None


def test_derive_year_from_source_name():
    """year 从 source_name 正则提取。"""
    assert derive_year({"source_name": "小米集团2025年报"}) == 2025


def test_derive_year_none_when_no_year():
    assert derive_year({"source_name": "年报"}) is None
    assert derive_year({}) is None


def test_enrich_metadata_adds_missing_fields():
    """补 document_id + year，保留原有字段。"""
    meta = enrich_metadata(
        {"company_id": "xiaomi", "source": "小米2025年报.pdf",
         "source_name": "小米2025年报"}
    )
    assert meta["document_id"] == md5("小米2025年报.pdf".encode()).hexdigest()
    assert meta["year"] == 2025
    assert meta["company_id"] == "xiaomi"


def test_enrich_metadata_preserves_existing():
    """已有 document_id/year 不覆盖（幂等）。"""
    meta = enrich_metadata({"document_id": "keep", "year": 2024})
    assert meta["document_id"] == "keep"
    assert meta["year"] == 2024


def test_enrich_metadata_does_not_mutate_input():
    """不可变：入参 metadata 不被修改。"""
    orig = {"company_id": "xiaomi", "source": "小米2025年报.pdf"}
    enrich_metadata(orig)
    assert "document_id" not in orig
    assert "year" not in orig


def test_enrich_record_reuses_embedding():
    """enrich_record 复用原 embedding（不重新 embedding）。"""
    rec = _make_record("内容", "c0")
    enriched = enrich_record(rec)
    assert enriched.embedding == rec.embedding
    assert enriched.chunk_id == rec.chunk_id
    assert enriched.metadata["year"] == 2025
    assert enriched.metadata["document_id"]


# ── B. 纯函数：分批 ──────────────────────────────────────────────

def test_batch_iter_splits():
    """7 条按 3 分批 → [3, 3, 1]，chunk 顺序保持。"""
    records = [_make_record(f"内容 {i}", f"c{i}") for i in range(7)]
    batches = list(batch_iter(records, 3))
    assert [len(b) for b in batches] == [3, 3, 1]
    assert [b[0].chunk_id for b in batches] == ["c0", "c3", "c6"]


# ── C. 集成：迁移 + 校验 + 回滚（真实 FAISSStore + fake MilvusStore）────

def test_migrate_company_export_insert_batches():
    """5 条分批 2 → exported=5/inserted=5/batches=3，Milvus 里是补全后行。"""
    faiss = _faiss_store()
    faiss.add([_make_record(f"内容 {i}", f"c{i}") for i in range(5)])
    milvus = _fake_milvus()

    report = migrate_company(faiss, milvus, "xiaomi", batch_size=2)

    assert report["exported"] == 5
    assert report["inserted"] == 5
    assert report["batches"] == 3
    rows = _milvus_rows(milvus, "xiaomi")
    assert len(rows) == 5
    assert all(r["metadata"]["document_id"] for r in rows)
    assert all(r["metadata"]["year"] == 2025 for r in rows)


def test_migrate_company_empty_faiss():
    """FAISS 空库 → 0 插入，不崩。"""
    faiss = _faiss_store()
    milvus = _fake_milvus()
    report = migrate_company(faiss, milvus, "xiaomi")
    assert report == {"company": "xiaomi", "exported": 0, "inserted": 0, "batches": 0}


def test_validate_migration_pass_after_migrate():
    """迁移后校验全绿（count + 抽样 cosine>=0.9999 + 元数据一致）。"""
    faiss = _faiss_store()
    faiss.add([_make_record(f"内容 {i}", f"c{i}") for i in range(4)])
    milvus = _fake_milvus()
    migrate_company(faiss, milvus, "xiaomi")

    result = validate_migration(faiss, milvus, "xiaomi", sample_size=20)

    assert result["ok"] is True
    assert result["faiss_count"] == 4
    assert result["milvus_count"] == 4
    assert result["sample_checked"] == 4
    assert result["issues"] == []


def test_validate_detects_count_mismatch():
    """未迁移（Milvus 空）→ count mismatch。"""
    faiss = _faiss_store()
    faiss.add([_make_record("内容", "c0")])
    milvus = _fake_milvus()

    result = validate_migration(faiss, milvus, "xiaomi")

    assert result["ok"] is False
    assert any("count mismatch" in i for i in result["issues"])


def test_validate_detects_embedding_mismatch():
    """篡改 Milvus embedding → cosine 低于 0.9999。"""
    faiss = _faiss_store()
    faiss.add([_make_record("内容", "c0")])
    milvus = _fake_milvus()
    migrate_company(faiss, milvus, "xiaomi")
    for row in milvus._client._collections["test_finance_knowledge"]["rows"]:
        if row["chunk_id"] == "c0":
            row["embedding"][0] += 1.0

    result = validate_migration(faiss, milvus, "xiaomi", sample_size=20)

    assert result["ok"] is False
    assert any("embedding mismatch" in i for i in result["issues"])


def test_validate_detects_text_mismatch():
    """篡改 Milvus text → text mismatch。"""
    faiss = _faiss_store()
    faiss.add([_make_record("内容", "c0")])
    milvus = _fake_milvus()
    migrate_company(faiss, milvus, "xiaomi")
    for row in milvus._client._collections["test_finance_knowledge"]["rows"]:
        if row["chunk_id"] == "c0":
            row["text"] = "被篡改的原文"

    result = validate_migration(faiss, milvus, "xiaomi", sample_size=20)

    assert result["ok"] is False
    assert any("text mismatch" in i for i in result["issues"])


def test_validate_detects_metadata_mismatch():
    """篡改 Milvus metadata（section）→ metadata[section] mismatch。"""
    faiss = _faiss_store()
    faiss.add([_make_record("内容", "c0")])
    milvus = _fake_milvus()
    migrate_company(faiss, milvus, "xiaomi")
    for row in milvus._client._collections["test_finance_knowledge"]["rows"]:
        if row["chunk_id"] == "c0":
            row["metadata"]["section"] = "被篡改"

    result = validate_migration(faiss, milvus, "xiaomi", sample_size=20)

    assert result["ok"] is False
    assert any("metadata[section]" in i for i in result["issues"])


def test_rollback_company_deletes_only_target():
    """回滚只删本公司数据，共享 collection 中他司数据保留。"""
    faiss = _faiss_store()
    faiss.add([_make_record("小米内容", "c0")])
    milvus = _fake_milvus()
    migrate_company(faiss, milvus, "xiaomi")
    # 模拟共享 collection 中其他公司数据
    milvus._client.upsert(
        "test_finance_knowledge",
        [{"chunk_id": "catl-0", "company_id": "catl", "text": "宁德内容",
          "embedding": DummyEmbeddingModel(dim=DIM).embed(["宁德内容"])[0],
          "metadata": {"company_id": "catl"}}],
    )

    removed = rollback_company(milvus, "xiaomi")

    assert removed == 1
    assert _milvus_rows(milvus, "xiaomi") == []
    assert len(_milvus_rows(milvus, "catl")) == 1
