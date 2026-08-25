"""PR44.3.2 迁移后一致性校验（独立运行；FAISS=source of truth, Milvus=replica）。

复用 ``scripts/migrate_faiss_to_milvus.py`` 的 ``validate_migration``：
  迁移前预检（Milvus 尚无数据 → 应报 count mismatch）/ 迁移后复查 / 任意时刻对账。

用法：
    PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe \\
        scripts/verify_milvus.py
        scripts/verify_milvus.py --company 小米 --sample 50
    （默认抽样 100；PR44.3.3 一致性验证要求：chunk_id/metadata/text + cosine>=0.9999）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows 控制台 GBK 兜底
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.rag.vectorstore import MilvusStore, get_store
from scripts.migrate_faiss_to_milvus import _discover_companies, validate_migration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FAISS vs Milvus 一致性校验")
    parser.add_argument("--company", default=None, help="只校验指定公司（默认全部）")
    parser.add_argument("--sample", type=int, default=100, help="抽样 chunk 数")
    args = parser.parse_args(argv)

    companies = [args.company] if args.company else _discover_companies()
    if not companies:
        print("未发现 FAISS 数据（data/vector_store/*/index.faiss）")
        return 1

    milvus = MilvusStore(
        dim=settings.milvus_dim,
        uri=settings.milvus_uri,
        collection_name=settings.milvus_collection_name,
    )
    all_ok = True
    for company in companies:
        faiss = get_store(company_id=company)
        result = validate_migration(faiss, milvus, company, sample_size=args.sample)
        status = "PASS" if result["ok"] else "FAIL"
        print(f"[{status}] {company}: FAISS={result['faiss_count']} "
              f"Milvus={result['milvus_count']} sample={result['sample_checked']}")
        for issue in result["issues"]:
            print(f"    ! {issue}")
        all_ok = all_ok and result["ok"]
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
