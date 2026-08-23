"""PR44.3.3 Milvus Integration Benchmark：FAISS vs Milvus（等价值验证）。

目标（用户拍板）：证明 **Milvus ≈ FAISS**（等价值），**非**"Milvus 替代 FAISS"。
同一 query 集跑两条 Dense→RRF→Rerank 全管线（复用生产 ``app.rag.retrieve``），对比：

  - Accuracy：Recall@1/5/10 + MRR + NDCG@5/10
    （门禁：Xiaomi R@5≥80% / MRR≥0.423 · CATL R@5=100% / MRR≥0.950）
  - Latency：per-query P50/P95/P99（全管线；embedding 两端共用，隔离后端差异）
  - Resource：FAISS 索引 ntotal/文件大小 vs Milvus collection stats/索引

数据两侧一一对应（AD-3 chunk_id 关联，verify_milvus.py 已对账）：
  - FAISS 侧  = data/vector_store/<公司>（生产归档）
  - Milvus 侧 = finance_agent.finance_knowledge（迁移副本）

用法：
    PYTHONIOENCODING=utf-8 /d/dev/conda/envs/finance-agent/python.exe \\
        scripts/benchmark_milvus.py                 # 全部公司
        scripts/benchmark_milvus.py --company 小米
        scripts/benchmark_milvus.py --repeat 5      # 每 query 跑 N 轮取中位（latency 降噪）

**不改 DEFAULT_VECTOR_BACKEND**（backend 仍 faiss）；生产切换是 PR44.4 独立任务。
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows 控制台 GBK 兜底 + 行缓冲（长任务可见进度）
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.rag.evaluation.dataset import load_dataset
from app.rag.evaluation.metrics import compute_metrics
from app.rag.vectorstore import clear_store_cache, get_store

_DATASETS = Path(__file__).resolve().parents[1] / "evaluation" / "datasets"

# 冻结门禁（用户 PR44.3.3 规格：keep Xiaomi R@5≥80%/MRR≥0.423, CATL R@5=100%/MRR≥0.950）
_GATES: dict[str, dict[str, float]] = {
    "小米": {"recall_at_k[5]": 0.80, "mrr": 0.423},
    "宁德时代": {"recall_at_k[5]": 1.00, "mrr": 0.950},
}
_TOP_K = 5


# ── 纯函数（可单测）──────────────────────────────────────────────


def hit_ranks(chunks: list, expected_sections: list[str]) -> list[int]:
    """返回 top-K 内命中期望章节的 rank（1-indexed，section/chapter 任一命中即算）。

    与 tests/eval_helpers.py 的命中判定一致：subtitle（section）可能覆盖 outline 章节名。
    """
    ranks: list[int] = []
    for i, c in enumerate(chunks):
        sec = c.metadata.get("section", "") or ""
        chap = c.metadata.get("chapter", "") or ""
        if any(exp in sec or exp in chap for exp in expected_sections):
            ranks.append(i + 1)
    return ranks


def percentile(sorted_ms: list[float], p: float) -> float:
    """线性插值 P 分位（rank = p/100*(n-1)，标准定义；小样本也比 nearest-rank 平滑）。"""
    if not sorted_ms:
        return 0.0
    n = len(sorted_ms)
    rank = (p / 100.0) * (n - 1)
    lo = math.floor(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_ms[lo] * (1 - frac) + sorted_ms[hi] * frac


def _check_gates(
    company: str, faiss_m, milvus_m, faiss_lat: list[float], milvus_lat: list[float]
) -> list[str]:
    """门禁 + 等价值断言。返回失败项列表（空 = 全过）。"""
    fails: list[str] = []
    gates = _GATES.get(company, {})
    for metric, floor in gates.items():
        if metric == "recall_at_k[5]":
            fv = faiss_m.recall_at_k.get(5, 0.0)
            mv = milvus_m.recall_at_k.get(5, 0.0)
        elif metric == "mrr":
            fv, mv = faiss_m.mrr, milvus_m.mrr
        else:
            continue
        if fv < floor:
            fails.append(f"FAISS {metric}={fv:.3f} < 门禁 {floor}")
        if mv < floor:
            fails.append(f"Milvus {metric}={mv:.3f} < 门禁 {floor}")
    # 等价值：Milvus 精度不显著低于 FAISS（容差 1 个百分点，避免噪声误报）
    for metric in ("recall_at_k[5]", "mrr", "ndcg_at_k[5]"):
        if metric == "recall_at_k[5]":
            fv, mv = faiss_m.recall_at_k.get(5, 0.0), milvus_m.recall_at_k.get(5, 0.0)
        elif metric == "mrr":
            fv, mv = faiss_m.mrr, milvus_m.mrr
        else:
            fv, mv = faiss_m.ndcg_at_k.get(5, 0.0), milvus_m.ndcg_at_k.get(5, 0.0)
        if mv < fv - 0.01:
            fails.append(f"Milvus {metric}={mv:.3f} 低于 FAISS {fv:.3f} 超容差 0.01")
    # Latency：Milvus P95 不明显慢于 FAISS（容差 3x，跨网络 gRPC 本就慢）
    if percentile(sorted(milvus_lat), 95) > percentile(sorted(faiss_lat), 95) * 3:
        fails.append("Milvus P95 latency > 3x FAISS P95")
    return fails


# ── 资源统计 ─────────────────────────────────────────────────────


def _faiss_resource(company: str) -> dict:
    base = Path(settings.rag_vector_store_path) / company
    idx, meta = base / "index.faiss", base / "metadata.json"
    return {
        "index_size_kb": round(idx.stat().st_size / 1024, 1) if idx.exists() else 0,
        "metadata_size_kb": round(meta.stat().st_size / 1024, 1) if meta.exists() else 0,
    }


def _milvus_resource(store) -> dict:
    client = store._client  # 白盒：一次性运维脚本直接读 client（同 migrate/verify）
    out: dict = {"row_count": None, "indexes": []}
    try:
        stats = client.get_collection_stats(store._collection)
        out["row_count"] = stats.get("row_count")
    except Exception:  # noqa: BLE001, S110  # 一次性运维：资源探测失败不阻断 benchmark
        pass
    try:
        # list_indexes 返回索引名（字段名）；describe_index 才含 FLAT/COSINE 详情（AD-4）
        index_names = list(client.list_indexes(store._collection))
        if index_names:
            desc = client.describe_index(store._collection, index_names[0])
            out["indexes"] = [
                f"{desc.get('field_name')}:{desc.get('index_type')}/{desc.get('metric_type')}"
            ]
    except Exception:  # noqa: BLE001, S110
        pass
    return out


# ── 主流程 ───────────────────────────────────────────────────────


def _resolve_companies(arg: str | None) -> list[tuple[str, str]]:
    """解析要测的公司 → [(dataset_stem, company_display)]。

    --company 可传中文名（小米/宁德时代）或数据集 stem（xiaomi/catl）；
    默认扫描 evaluation/datasets/*.json 全部。
    """
    stems = sorted(d.stem for d in _DATASETS.iterdir() if d.suffix == ".json")
    ds_by_stem = {s: load_dataset(str(_DATASETS / (s + ".json"))) for s in stems}
    if not arg:
        return [(s, ds_by_stem[s].company) for s in stems]
    if arg in ds_by_stem:
        return [(arg, ds_by_stem[arg].company)]
    for s, ds in ds_by_stem.items():
        if ds.company == arg:
            return [(s, arg)]
    raise SystemExit(f"找不到 {arg} 对应数据集（evaluation/datasets/*.json）")


def _run_company(dataset_stem: str, company: str, repeat: int) -> int:
    """单公司 benchmark。返回 0 = 门禁通过。"""
    ds = load_dataset(str(_DATASETS / (dataset_stem + ".json")))
    items = [it for it in ds.items if it.company == company] or ds.items
    print(f"\n===== {company}（{len(items)} queries）=====")

    clear_store_cache()
    # FAISS 侧显式钉死（PR44.4：默认后端已配置化，不显式会随生产配置漂移导致对比失效）
    faiss_store = get_store(company_id=company, backend="faiss")
    milvus_store = get_store(company_id=company, backend="milvus")

    from app.rag import retrieve

    def _run(store) -> tuple[list[list[int]], list[float]]:
        hit_ranks_all: list[list[int]] = []
        lat: list[float] = []
        for item in items:
            rounds: list[float] = []
            for _ in range(repeat):
                t0 = time.perf_counter()
                res = retrieve(item.query, company=company, top_k=_TOP_K, _store=store)
                rounds.append((time.perf_counter() - t0) * 1000.0)
            lat.append(statistics.median(rounds))
            hit_ranks_all.append(hit_ranks(res.chunks, item.expected_sections))
        return hit_ranks_all, lat

    # 预热（触发模型加载 + BM25 语料构建 + reranker 加载，不计入稳态计时）
    _run(faiss_store)
    _run(milvus_store)

    faiss_ranks, faiss_lat = _run(faiss_store)
    milvus_ranks, milvus_lat = _run(milvus_store)
    n = len(items)

    faiss_m = compute_metrics(faiss_ranks, n_queries=n)
    milvus_m = compute_metrics(milvus_ranks, n_queries=n)

    def _line(label: str, m) -> str:
        return (
            f"{label:9s} R@1={m.recall_at_k.get(1,0):.3f} R@5={m.recall_at_k.get(5,0):.3f} "
            f"R@10={m.recall_at_k.get(10,0):.3f} MRR={m.mrr:.3f} NDCG@5={m.ndcg_at_k.get(5,0):.3f}"
        )

    print(_line("FAISS", faiss_m))
    print(_line("Milvus", milvus_m))
    print(
        f"Latency(ms) FAISS  P50={percentile(sorted(faiss_lat),50):.0f} "
        f"P95={percentile(sorted(faiss_lat),95):.0f} P99={percentile(sorted(faiss_lat),99):.0f}"
    )
    print(
        f"Latency(ms) Milvus P50={percentile(sorted(milvus_lat),50):.0f} "
        f"P95={percentile(sorted(milvus_lat),95):.0f} P99={percentile(sorted(milvus_lat),99):.0f}"
    )
    print("Resource FAISS :", _faiss_resource(company))
    print("Resource Milvus:", _milvus_resource(milvus_store))

    fails = _check_gates(company, faiss_m, milvus_m, faiss_lat, milvus_lat)
    status = "PASS" if not fails else "FAIL"
    print(f"[{status}] {company} 门禁+等价值")
    for f in fails:
        print(f"    ! {f}")
    return 0 if not fails else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FAISS vs Milvus 检索 Benchmark")
    parser.add_argument("--company", default=None, help="只测指定公司（默认全部）")
    parser.add_argument("--repeat", type=int, default=3, help="每 query 轮数（latency 取中位）")
    args = parser.parse_args(argv)

    # 对齐生产回归设置（PR #33 metadata 精排 + rule 改写），保证与既有基线可比
    settings.rag_query_rewriter = "rule"
    settings.rag_reranker_model = "metadata"
    settings.rag_metadata_company_weights = {"小米": "0.90,0.08,0.02"}
    try:  # CUDA 可用时 embedding/CE 走 GPU（回归评测同样处理，避免 CPU 数十倍慢）
        import torch

        if torch.cuda.is_available():
            settings.rag_embedding_device = "cuda"
    except ImportError:
        pass

    companies = _resolve_companies(args.company)
    exit_codes = [_run_company(stem, comp, args.repeat) for stem, comp in companies]
    return 1 if any(exit_codes) else 0


if __name__ == "__main__":
    sys.exit(main())
