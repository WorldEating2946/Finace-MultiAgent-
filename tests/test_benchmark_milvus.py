"""PR44.3.3 Benchmark 纯函数单测（离线，无需 Milvus / BGE）。

只测 benchmark 脚本的纯逻辑（hit_ranks / percentile）；
管线执行（retrieve + 真 Milvus）属一次性运维验证，不在单测范围。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_milvus import hit_ranks, percentile

# 命中判定依赖 DocumentChunk 的 metadata（section/chapter），用轻量对象替代。


class _FakeChunk:
    def __init__(self, metadata: dict):
        self.metadata = metadata


# ── hit_ranks ────────────────────────────────────────────────────

def test_hit_ranks_section_match():
    """section 命中期望章节 → 记录 rank。"""
    chunks = [
        _FakeChunk({"section": "管理层讨论及分析"}),
        _FakeChunk({"section": "财务报表"}),
        _FakeChunk({"section": "环境社会及管治"}),
    ]
    assert hit_ranks(chunks, ["财务报表"]) == [2]


def test_hit_ranks_chapter_fallback():
    """section 缺失时回退 chapter。"""
    chunks = [
        _FakeChunk({"chapter": "董事会报告"}),
        _FakeChunk({"section": "其他"}),
    ]
    assert hit_ranks(chunks, ["董事会报告"]) == [1]


def test_hit_ranks_multiple_and_empty():
    """多命中记录全部 rank；无命中返回空。"""
    chunks = [_FakeChunk({"section": "A"}), _FakeChunk({"section": "B"})]
    assert hit_ranks(chunks, ["A"]) == [1]
    assert hit_ranks(chunks, ["不存在"]) == []


# ── percentile ───────────────────────────────────────────────────

def test_percentile_basic():
    """线性插值 P50/P95/P99（rank=p/100*(n-1)）。"""
    ms = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    assert percentile(ms, 50) == 55.0
    assert abs(percentile(ms, 95) - 95.5) < 1e-9
    assert abs(percentile(ms, 99) - 99.1) < 1e-9


def test_percentile_empty():
    """空样本返回 0，不崩。"""
    assert percentile([], 95) == 0.0
