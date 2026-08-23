"""MilvusStore 测试用 in-memory fake（实现 pymilvus.MilvusClient 被用子集）。

PR44.3.1 测试策略：Windows 不支持 Milvus Lite，且适配器测试不应依赖真实 Milvus 服务。
本 fake 用 numpy 余弦相似度暴力扫描 = Milvus FLAT/COSINE 的精确结果（AD-4），
expr 求值器只覆盖 MilvusStore._build_expr 生成的语法 + delete 的 ``in [...]``。

不依赖 pymilvus——``tests/`` 包内模块（对应用例见 tests/test_milvus_adapter.py）。
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# ── 表达式求值（MilvusStore._build_expr 输出子集 + in [...]）─────────


def _eval_expr(row: dict, expr: str) -> bool:
    """求值 Milvus 风格布尔表达式（空串视为 True）。"""
    if not expr or not expr.strip():
        return True
    for clause in _split_and(expr):
        clause = clause.strip()
        if clause and not _eval_clause(row, clause):
            return False
    return True


def _split_and(expr: str) -> list[str]:
    """按顶层 `` and `` 切分，忽略引号内与方括号内的 and。"""
    parts: list[str] = []
    depth = 0
    in_quote: str | None = None
    cur: list[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if in_quote:
            cur.append(ch)
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_quote = ch
            cur.append(ch)
        elif ch == "[":
            depth += 1
            cur.append(ch)
        elif ch == "]":
            depth -= 1
            cur.append(ch)
        elif depth == 0 and expr.startswith(" and ", i):
            parts.append("".join(cur))
            cur = []
            i += 5
            continue
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return parts


def _resolve_lhs(row: dict, token: str) -> Any:
    """解析表达式左侧：metadata["key"] → 嵌套 JSON；否则顶层字段。"""
    m = re.match(r'^(?:\$meta|metadata)\s*\[\s*["\'](.+?)["\']\s*\]\s*$', token)
    if m:
        return (row.get("metadata") or {}).get(m.group(1))
    return row.get(token)


def _parse_rhs(token: str) -> Any:
    """解析表达式右侧：引号字符串 / 整数 / 浮点。"""
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ('"', "'"):
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _eval_clause(row: dict, clause: str) -> bool:
    # in [...]（delete 用：chunk_id in ["a", "b"]）
    m = re.match(r'^(.+?)\s+in\s+\[(.*)\]\s*$', clause)
    if m:
        values = [_parse_rhs(v) for v in m.group(2).split(",")]
        return _resolve_lhs(row, m.group(1).strip()) in values
    # 比较运算符（长到短，避免 == 先于 >= 匹配）
    for op in (">=", "<=", "!=", "==", ">", "<"):
        if op in clause:
            lhs_s, _, rhs_s = clause.partition(op)
            lhs = _resolve_lhs(row, lhs_s.strip())
            rhs = _parse_rhs(rhs_s.strip())
            if isinstance(rhs, (int, float)):
                try:
                    return _compare(float(lhs), op, float(rhs))
                except (TypeError, ValueError):
                    pass
            return _compare(str(lhs), op, str(rhs))
    return True


def _compare(a: Any, op: str, b: Any) -> bool:
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op == ">=":
        return a >= b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == "<":
        return a < b
    return False


# ── 向量相似度 ────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _select(row: dict, output_fields: list[str] | None) -> dict:
    """按 output_fields 选取；None/空 → 全部（排除 embedding 避免超大返回）。"""
    if not output_fields:
        return {k: v for k, v in row.items() if k != "embedding"}
    return {f: row.get(f) for f in output_fields}


# ── MilvusClient 子集 fake ───────────────────────────────────────


class ContractFakeMilvusClient:
    """内存实现 MilvusStore 用到的 pymilvus.MilvusClient 方法子集。

    collection 结构：{"dimension", "metric_type", "rows": list[dict]}，
    rows 为 upsert 的 dict（chunk_id 主键，含 embedding / 标量字段 / metadata JSON）。
    """

    _CONTRACT_FAKE = True  # MilvusStore 用它区分 fake 与真实 client（_ensure_collection）

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, Any]] = {}

    # ── 服务/库级探测（PR44.4 健康检查用，对齐真实 env v2.4.0）──────

    def get_server_version(self) -> str:
        return "v2.4.0"

    def list_databases(self) -> list[str]:
        # 默认对齐真实 env：default + finance_agent；测试可改 self._databases 模拟缺库
        return list(getattr(self, "_databases", ["default", "finance_agent"]))

    def describe_collection(self, collection_name: str) -> dict:
        col = self._collections[collection_name]
        return {"fields": [{"name": "embedding", "params": {"dim": col["dimension"]}}]}

    # ── collection 生命周期 ─────────────────────────────────────

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self._collections

    def drop_collection(self, collection_name: str) -> None:
        self._collections.pop(collection_name, None)

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric_type: str = "COSINE",
        **kwargs: Any,
    ) -> None:
        self._collections[collection_name] = {
            "dimension": dimension,
            "metric_type": metric_type,
            "rows": [],
        }

    def get_collection_stats(self, collection_name: str) -> dict:
        return {"row_count": len(self._collections[collection_name]["rows"])}

    # ── 数据操作 ────────────────────────────────────────────────

    def upsert(self, collection_name: str, data: list[dict]) -> dict:
        col = self._collections[collection_name]
        rows = {r["chunk_id"]: r for r in col["rows"]}
        for row in data:
            rows[row["chunk_id"]] = row
        col["rows"] = list(rows.values())
        return {"upsert_count": len(data)}

    def delete(self, collection_name: str, filter: str) -> dict:
        col = self._collections[collection_name]
        keep = [r for r in col["rows"] if not _eval_expr(r, filter)]
        removed = len(col["rows"]) - len(keep)
        col["rows"] = keep
        return {"delete_count": removed}

    def query(
        self,
        collection_name: str,
        filter: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        col = self._collections[collection_name]
        rows = [r for r in col["rows"] if not filter or _eval_expr(r, filter)]
        if output_fields == ["count(*)"]:
            return [{"count(*)": len(rows)}]
        return [_select(r, output_fields) for r in rows]

    def search(
        self,
        collection_name: str,
        data: list[list[float]],
        anns_field: str = "embedding",
        filter: str = "",
        output_fields: list[str] | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[list[dict]]:
        col = self._collections[collection_name]
        out: list[list[dict]] = []
        for query_vec in data:
            hits: list[dict] = []
            for row in col["rows"]:
                if filter and not _eval_expr(row, filter):
                    continue
                score = _cosine_similarity(query_vec, row[anns_field])
                hits.append(
                    {
                        "id": row.get("chunk_id", ""),
                        "distance": score,
                        "entity": _select(row, output_fields),
                    }
                )
            hits.sort(key=lambda h: h["distance"], reverse=True)
            out.append(hits[:limit])
        return out
