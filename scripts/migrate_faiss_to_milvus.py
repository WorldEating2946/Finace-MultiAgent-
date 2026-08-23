"""PR44.3.2 FAISS → Milvus 一次性迁移工具。

定位（用户拍板）：Migration 是一次性运维任务，不是 runtime 能力 → 放 ``scripts/``，
**禁止** ``app/rag/vectorstore/migration.py``。

原则：
  - FAISS = source of truth，Milvus = migrated replica。本工具只写 Milvus，不动 FAISS；
  - 复用 FAISS 现有 embedding，**不重新 embedding**（校验 cosine >= 0.9999）；
  - 元数据补全（前置核查 Q2）：document_id=md5(source)、year=source_name 正则提取、
    document_type 保持 ``source_type`` JSON 别名（已拍板）、section 允许空；
  - 幂等：MilvusStore.add() = upsert（chunk_id 主键），重复运行安全。

用法（需已启动 Milvus 服务，见 docs/pr44_milvus_design.md）：
    PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe \\
        scripts/migrate_faiss_to_milvus.py                      # 迁移全部公司
        scripts/migrate_faiss_to_milvus.py --company 小米 --dry-run
        scripts/migrate_faiss_to_milvus.py --rollback           # 清空 Milvus（不动 FAISS）

可测试性：核心函数接收 store 对象（FAISSStore / MilvusStore），单测注入
ContractFakeMilvusClient 离线验证（见 tests/test_migration_helpers.py）。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from hashlib import md5
from pathlib import Path
from re import search

if hasattr(sys.stdout, "reconfigure"):  # Windows 控制台 GBK 兜底
    sys.stdout.reconfigure(encoding="utf-8")

# 允许 `python scripts/migrate_faiss_to_milvus.py` 直接运行（app 包在上级目录）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.core.config import settings
from app.rag.vectorstore import MilvusStore, VectorRecord, VectorStore, get_store

_DEFAULT_BATCH_SIZE = 500
_COSINE_THRESHOLD = 0.9999
# 迁移后需与 FAISS 一致的元数据字段（document_id/year 由补全步骤产生）。
_META_CHECK_FIELDS = ("company_id", "source_type", "year", "section")


# ── 元数据补全（前置核查 Q2：FAISS 缺 document_id / year）────────────────


def derive_document_id(metadata: dict) -> str | None:
    """document_id = md5(source)（每文档一个 hash）；source 缺失退 source_name。"""
    source = metadata.get("source") or metadata.get("source_name")
    if not source:
        return None
    return md5(str(source).encode("utf-8")).hexdigest()


def derive_year(metadata: dict) -> int | None:
    """year 从 source_name/source 正则提取（"小米集团2025年报" → 2025）。"""
    name = metadata.get("source_name") or metadata.get("source") or ""
    m = search(r"(20\d{2})", str(name))
    return int(m.group(1)) if m else None


def enrich_metadata(metadata: dict) -> dict:
    """补全元数据（缺 document_id / year 时派生），返回新 dict，不修改入参。"""
    meta = dict(metadata)
    if not meta.get("document_id"):
        did = derive_document_id(meta)
        if did:
            meta["document_id"] = did
    if not meta.get("year"):
        year = derive_year(meta)
        if year is not None:
            meta["year"] = year
    return meta


def enrich_record(record: VectorRecord) -> VectorRecord:
    """返回补全后的新 VectorRecord（embedding 原样复用，不重新 embedding）。"""
    return VectorRecord(
        chunk_id=record.chunk_id,
        text=record.text,
        embedding=record.embedding,
        metadata=enrich_metadata(record.metadata),
    )


def batch_iter(
    records: list[VectorRecord], batch_size: int
) -> Iterator[list[VectorRecord]]:
    """按 batch_size 分批（最后一批可能不足）。"""
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


# ── 迁移执行 ─────────────────────────────────────────────────────


def migrate_company(
    faiss: VectorStore,
    milvus: MilvusStore,
    company: str,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> dict:
    """单公司迁移：读 FAISS → 补全 → 分批 add 到 Milvus。

    Args:
        faiss:    FAISSStore（source of truth，all_chunks 读全部活跃 chunk）。
        milvus:   MilvusStore（replica，add=upsert 幂等）。
        company:  公司标识（用于日志；chunk 自身的 company_id 来自 metadata）。

    Returns:
        {"company", "exported", "inserted", "batches"}。
    """
    chunks = faiss.all_chunks()
    if not chunks:
        return {"company": company, "exported": 0, "inserted": 0, "batches": 0}
    enriched = [enrich_record(c) for c in chunks]
    inserted = 0
    batches = 0
    for batch in batch_iter(enriched, batch_size):
        milvus.add(batch)
        inserted += len(batch)
        batches += 1
    return {
        "company": company,
        "exported": len(chunks),
        "inserted": inserted,
        "batches": batches,
    }


# ── 一致性校验（FAISS = source of truth，Milvus = replica）────────────


def _milvus_rows(milvus: MilvusStore, company: str) -> list[dict]:
    """读 Milvus 中某公司全量行（chunk_id/embedding/metadata）。

    白盒访问：MilvusStore 未暴露 per-company count/vector 查询，一次性运维
    脚本直接走底层 client.query（真实与 fake client 签名一致）。
    """
    client = milvus._client
    collection = milvus._collection
    if not client.has_collection(collection):
        return []
    return client.query(
        collection_name=collection,
        filter=f'company_id == "{company}"',
        output_fields=["chunk_id", "embedding", "metadata", "text"],
    )


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def validate_migration(
    faiss: VectorStore,
    milvus: MilvusStore,
    company: str,
    sample_size: int = 20,
) -> dict:
    """FAISS vs Milvus 一致性校验。

    校验项：
      1. count 一致（FAISS 活跃 chunk 数 == Milvus 该公司行数）；
      2. 抽样 chunk 的 embedding cosine >= 0.9999（向量复用，非重算）；
      3. 抽样 chunk 原文 text 一致（FAISS 与 Milvus 逐字节）；
      4. 抽样 chunk 元数据一致（对 FAISS 元数据先补全再比较——迁移写入的是补全后行）。

    Returns:
        {"company", "faiss_count", "milvus_count", "sample_checked",
         "issues": list[str], "ok": bool}。
    """
    faiss_chunks = faiss.all_chunks()
    faiss_by_id = {c.chunk_id: c for c in faiss_chunks}
    milvus_rows = _milvus_rows(milvus, company)
    milvus_by_id = {r["chunk_id"]: r for r in milvus_rows}

    issues: list[str] = []
    if len(faiss_chunks) != len(milvus_rows):
        issues.append(
            f"count mismatch: FAISS={len(faiss_chunks)} Milvus={len(milvus_rows)}"
        )

    checked = 0
    for cid in list(faiss_by_id)[:sample_size]:
        fc = faiss_by_id[cid]
        mr = milvus_by_id.get(cid)
        if mr is None:
            issues.append(f"missing in Milvus: {cid}")
            continue
        checked += 1
        cos = _cosine(fc.embedding, mr.get("embedding") or [])
        if cos < _COSINE_THRESHOLD:
            issues.append(f"embedding mismatch {cid}: cos={cos:.6f}")
        if mr.get("text") != fc.text:
            issues.append(f"text mismatch {cid}: FAISS={fc.text[:40]!r} Milvus={mr.get('text','')[:40]!r}")
        # 迁移写入的是补全后行 → 用补全后的 FAISS 元数据对比
        faiss_meta = enrich_metadata(fc.metadata)
        milvus_meta = mr.get("metadata") or {}
        for key in _META_CHECK_FIELDS:
            if faiss_meta.get(key) != milvus_meta.get(key):
                issues.append(
                    f"metadata[{key}] mismatch {cid}: "
                    f"FAISS={faiss_meta.get(key)!r} Milvus={milvus_meta.get(key)!r}"
                )

    return {
        "company": company,
        "faiss_count": len(faiss_chunks),
        "milvus_count": len(milvus_rows),
        "sample_checked": checked,
        "issues": issues,
        "ok": not issues,
    }


# ── 回滚策略（FAISS 不动，只清 Milvus 中该公司数据）──────────────────


def rollback_company(milvus: MilvusStore, company: str) -> int:
    """删除 Milvus 中该公司全部行（幂等）。返回删除数。"""
    client = milvus._client
    collection = milvus._collection
    if not client.has_collection(collection):
        return 0
    result = client.delete(
        collection_name=collection, filter=f'company_id == "{company}"'
    )
    return int(result.get("delete_count", 0))


# ── CLI ──────────────────────────────────────────────────────────


def _discover_companies() -> list[str]:
    """扫描 data/vector_store/*/index.faiss 得到全部公司。"""
    root = Path(settings.rag_vector_store_path)
    if not root.exists():
        return []
    return sorted(
        d.name for d in root.iterdir() if d.is_dir() and (d / "index.faiss").exists()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FAISS → Milvus 一次性迁移工具")
    parser.add_argument("--company", default=None, help="只处理指定公司（默认全部）")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="只导出+补全报告，不写 Milvus")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--rollback", action="store_true", help="清空 Milvus 中目标公司数据")
    args = parser.parse_args(argv)

    companies = [args.company] if args.company else _discover_companies()
    if not companies:
        print("未发现可迁移的 FAISS 数据（data/vector_store/*/index.faiss）")
        return 1

    milvus = MilvusStore(
        dim=settings.milvus_dim,
        uri=settings.milvus_uri,
        collection_name=settings.milvus_collection_name,
    )

    if args.rollback:
        for company in companies:
            removed = rollback_company(milvus, company)
            print(f"[rollback] {company}: 删除 {removed} 行")
        return 0

    for company in companies:
        faiss = get_store(company_id=company)  # FAISS 后端加载现有索引
        chunks = faiss.all_chunks()
        print(f"[export] {company}: FAISS 活跃 chunk = {len(chunks)}")
        if args.dry_run:
            enriched = [enrich_record(c) for c in chunks]
            missing = sum(1 for c in enriched if not c.metadata.get("document_id"))
            yearless = sum(1 for c in enriched if not c.metadata.get("year"))
            print(f"[dry-run] 补全后：缺 document_id={missing} 缺 year={yearless}")
            continue
        report = migrate_company(faiss, milvus, company, batch_size=args.batch_size)
        print(f"[migrate] {company}: inserted={report['inserted']} "
              f"batches={report['batches']}")
        if not args.skip_validation:
            result = validate_migration(faiss, milvus, company)
            status = "PASS" if result["ok"] else "FAIL"
            print(f"[validate] {company}: FAISS={result['faiss_count']} "
                  f"Milvus={result['milvus_count']} "
                  f"sample={result['sample_checked']} -> {status}")
            for issue in result["issues"]:
                print(f"    ! {issue}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
