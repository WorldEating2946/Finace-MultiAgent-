"""Query Rewrite：弥合用户口语查询与年报正式文本之间的词汇鸿沟。

Phase 1：Rule-based（无 LLM 成本、可测试、可控）。
将单条查询扩展为多条语义变体（同义词替换），供 retriever 做 multi-query 召回。

原则：
    - 只放年会报告中真实存在的词汇鸿沟，不放通用词（避免过度扩展）；
    - 无匹配时返回 [query]（pass-through），保证精确查询（如 CATL）零影响。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

# 同义词表：键 = 用户口语/常见查询词，值 = 年报正式用语（按相关性降序）。
# 键为短语（非单字），降低子串误匹配；同义词已包含在查询中时跳过（防重复叠加）。
_SYNONYM_MAP: dict[str, list[str]] = {
    # 产品/业务 → 年报正式用语
    "汽车业务": ["智能电动汽车业务", "新能源汽车业务"],
    "汽车": ["智能电动汽车", "新能源汽车"],
    "智能手机": ["智能终端", "移动通信终端"],
    "IoT": ["物联网", "智能家居"],
    # 财报术语
    "研发投入": ["研发费用", "研发支出", "研发人员"],
    # 年报正文多用"收入"（小米 MD&A 实测 0 处"营业收入"），故首选"收入"
    "营业收入": ["收入", "主营业务收入", "营业总收入"],
    "全球出货量": ["全球销量", "全球交付量"],
    "出货量": ["销量", "交付量"],
    "经营风险": ["风险因素", "风险提示"],
    "股东权益": ["所有者权益", "归属于母公司所有者权益"],
    "现金": ["现金及现金等价物", "货币资金"],
    # 公司结构
    "公司治理": ["企业管治", "公司治理架构"],
    "治理结构": ["企业管治", "公司治理架构"],
    "董事会": ["董事及高级管理层", "董事会报告"],
}

# 单次 rewrite 最多返回的查询数（含原始）
_MAX_QUERIES = 4


class QueryRewriter(ABC):
    """查询重写抽象接口。"""

    @abstractmethod
    def rewrite(self, query: str) -> list[str]:
        """将单条查询扩展为多条变体。

        Returns:
            至少含原始查询的变体列表；无匹配时返回 [query]（pass-through）。
        """
        raise NotImplementedError


class RuleBasedQueryRewriter(QueryRewriter):
    """基于年报术语同义词表的查询扩展器（无 LLM 依赖）。

    匹配策略：
        - 键按长度降序匹配（"汽车业务" 优先于 "汽车"），避免子串互相覆盖；
        - 每个键最多贡献一个变体；同义词已包含在查询中则跳过；
        - 变体去重，总数上限 _MAX_QUERIES（含原始）。
    """

    def __init__(
        self,
        synonym_map: dict[str, list[str]] | None = None,
        max_queries: int = _MAX_QUERIES,
    ) -> None:
        # 注意：空 dict 是合法的"直通"配置，不能用 `or` 回退到默认表
        self._map = _SYNONYM_MAP if synonym_map is None else synonym_map
        self._max = max_queries
        # 键按长度降序：长键（更具体）先匹配
        self._keys = sorted(self._map, key=len, reverse=True)

    def rewrite(self, query: str) -> list[str]:
        results = [query]
        for key in self._keys:
            if key not in query:
                continue
            if len(results) >= self._max:
                break
            for syn in self._map[key]:
                if syn in query:
                    continue  # 同义词已存在（如"现金及现金等价物"含"现金"），无需扩展
                expanded = query.replace(key, syn, 1)
                if expanded != query and expanded not in results:
                    results.append(expanded)
                    break  # 每个键只贡献一个变体
        return results
