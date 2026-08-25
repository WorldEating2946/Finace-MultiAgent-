"""跨源冲突检测（PR #35）。

规则驱动（零 LLM 成本）：对每个画像维度，跨源匹配同名实体，比较描述相似度，
相似度低于阈值 → 记录 SourceConflict。

局限：启发式匹配无法覆盖所有命名差异；conflict 是"待人工 / Research Agent 判断"的
信号，不是自动裁决。

实体匹配与描述相似度同时被 fusion.py 复用（fused 画像合并同名实体）。
"""

from __future__ import annotations

import jieba

from app.rag.profile.schema import CompanyProfile
from app.rag.source.schema import FUSION_DIMENSIONS, SourceConflict

# 描述相似度低于该阈值 → 标记为冲突（"公司持续推进智能电动汽车业务" vs
# "汽车业务短期亏损压力较大" 的 jieba Jaccard ≈ 0.18 < 0.3）
_DESC_SIM_THRESHOLD = 0.3
# 实体名匹配：字符 bigram Jaccard 阈值
_NAME_BIGRAM_THRESHOLD = 0.4

# 实体名的常见业务后缀（跨源命名差异主要来源：研报省略"业务/事业部"等）
_NAME_SUFFIXES = ("业务", "事业部", "部门", "板块", "产业", "分部")


def _normalize_name(name: str) -> str:
    """去掉常见业务后缀，便于跨源匹配（"智能电动汽车业务" → "智能电动汽车"）。"""
    for suffix in _NAME_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)].strip()
    return name


def _char_bigrams(text: str) -> set[str]:
    """字符 bigram 集合（中文短名匹配比 jieba 分词更稳）。"""
    text = text.replace(" ", "").replace("，", "").replace("。", "")
    return {text[i : i + 2] for i in range(len(text) - 1)} if len(text) >= 2 else {text}


def _name_tokens(name: str) -> set[str]:
    """jieba 分词（仅保留长度≥2 的核心词，过滤单字虚词）。"""
    return set(t for t in jieba.lcut(_normalize_name(name)) if len(t) >= 2)


def same_entity(name_a: str, name_b: str) -> bool:
    """跨源实体匹配：同名 / 子串包含 / bigram Jaccard / 共享核心词，任一命中即视为同一实体。

    - "智能电动汽车业务" vs "汽车业务" → 子串包含（"汽车" ⊂ "智能电动汽车"）→ True；
    - "智能手机" vs "新能源汽车" → 无子串、无共享 → False。
    """
    na, nb = _normalize_name(name_a), _normalize_name(name_b)
    if na == nb:
        return True
    # 子串包含：jieba 可能把"电动汽车"合成一词，导致共享核心词漏判（"电动汽车" vs "汽车"）
    if len(na) >= 2 and len(nb) >= 2 and (na in nb or nb in na):
        return True
    ba, bb = _char_bigrams(na), _char_bigrams(nb)
    if ba and bb and len(ba & bb) / len(ba | bb) >= _NAME_BIGRAM_THRESHOLD:
        return True
    ta, tb = _name_tokens(na), _name_tokens(nb)
    return bool(ta & tb)


def _tokens(text: str) -> set[str]:
    return set(jieba.lcut(text))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def description_similarity(desc_a: str, desc_b: str) -> float:
    """描述相似度：jieba token Jaccard。"""
    return _jaccard(_tokens(desc_a), _tokens(desc_b))


class ConflictDetector:
    """跨源冲突检测：逐维度比较各源画像，识别矛盾描述。"""

    def detect(self, profiles: dict[str, CompanyProfile]) -> list[SourceConflict]:
        """对每对 source 组合、每个维度，比较匹配实体的描述。

        Args:
            profiles: {source_type: CompanyProfile}。

        Returns:
            冲突列表（无冲突 = 空列表）。
        """
        conflicts: list[SourceConflict] = []
        sources = list(profiles)
        for i, st_a in enumerate(sources):
            for st_b in sources[i + 1 :]:
                conflicts.extend(self._compare_pair(profiles, st_a, st_b))
        return conflicts

    def _compare_pair(
        self,
        profiles: dict[str, CompanyProfile],
        st_a: str,
        st_b: str,
    ) -> list[SourceConflict]:
        pa, pb = profiles[st_a], profiles[st_b]
        conflicts: list[SourceConflict] = []
        for dim in FUSION_DIMENSIONS:
            for ia in getattr(pa, dim):
                for ib in getattr(pb, dim):
                    if not same_entity(ia.name, ib.name):
                        continue
                    if description_similarity(ia.description, ib.description) < _DESC_SIM_THRESHOLD:
                        conflicts.append(
                            SourceConflict(
                                dimension=dim,
                                entity_a=ia.name,
                                entity_b=ib.name,
                                claim_a=ia.description,
                                claim_b=ib.description,
                                source_a=st_a,
                                source_b=st_b,
                                evidence_a=ia.evidence[0] if ia.evidence else None,
                                evidence_b=ib.evidence[0] if ib.evidence else None,
                                resolution_note=(
                                    f"{st_a} 与 {st_b} 对「{ia.name}」描述不一致，"
                                    "需结合证据链人工 / Research Agent 判断。"
                                ),
                            )
                        )
        return conflicts
