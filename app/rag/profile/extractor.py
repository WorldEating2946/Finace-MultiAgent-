"""企业知识画像提取器（PR #34）。

数据流：RAG 召回目标字段的 chunks → LLM 结构化抽取 → 证据归因（Evidence Chain）。

抗幻觉设计：
    - LLM 只返回 chunk 索引号（evidence_refs），source/page/chapter 由后处理从
      已知 metadata 补全 —— LLM 无法编造章节；
    - evidence_refs 越界索引被静默丢弃；
    - quote 取自 LLM 提供的原文，若与 chunk 文本不符则回退为 chunk 文本片段（仍是真实原文）。

LLM 模式：DeepSeek（OpenAI 兼容），urllib 直连（零新增依赖，复用 llm_rewriter 模式）。
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone

from app.core.config import settings
from app.rag.profile.schema import CompanyProfile, EvidenceRef, ProfileItem

# 每个维度检索 query 模板
_FIELD_QUERIES: dict[str, str] = {
    "business_segments":      "{company} 主营业务 业务构成 分部",
    "products":               "{company} 主要产品 产品线",
    "technologies":           "{company} 核心技术 研发投入 技术",
    "customers":              "{company} 客户 用户 市场定位",
    "geographic_markets":     "{company} 海外市场 地理区域 全球化 全国",
    "competitive_advantages": "{company} 竞争优势 竞争壁垒 领先",
    "risks":                  "{company} 经营风险 风险因素",
    "strategic_direction":    "{company} 未来战略 发展战略 未来规划",
}

_MAX_ITEMS_PER_FIELD = 8
_MAX_QUOTE_CHARS = 200
_MAX_LLM_RETRIES = 3  # 空 content 重试次数（推理模型间歇性返回空）

_FIELD_PROMPT = """你是企业年报结构化抽取专家。从下列检索到的 {company} 年报文段中，提取「{field}」。

检索文段（每条带 [索引] 和来源标注）：
{context}

要求：
1. 只提取文段中**明确提到**的信息，不推断、不编造；文段未涉及的维度返回空数组 []；
2. 每个实体：name（实体名）、description（一句话，≤40 字）、evidence_refs（支撑该实体的文段索引号列表）；
3. quote 从对应文段中**逐字摘录**（≤{max_quote} 字符），不得改写；
4. 最多 {max_items} 个实体。

只输出 JSON 数组，不要任何其他文字，例如：
[{{"name": "智能手机", "description": "核心主业，全球前三", "evidence_refs": [0, 2], "quotes": ["2025年智能手机收入..."]}}]"""


class ProfileExtractor:
    """逐字段提取：RAG 检索 → LLM 结构化抽取 → 证据归因。"""

    def __init__(
        self,
        company: str,
        top_k: int = 5,
        source_type: str = "",
        *,
        _call: callable | None = None,
    ) -> None:
        """Args:
            company:      目标公司（向量库过滤维度）。
            top_k:        每维度召回的 chunk 数。
            source_type:  限定文档语义类型（PR #35）：annual_report / research_report /
                          policy / news；空串 = 所有源混检（默认，向后兼容）。
            _call:        测试 seam —— 注入 LLM 调用函数（prompt -> content）。
        """
        self._company = company
        self._top_k = top_k
        self._source_type = source_type
        self._call = _call or self._http_call

    # ── 单维度提取 ──────────────────────────────────────────────
    def extract_field(self, field_name: str, retrieval_query: str) -> list[ProfileItem]:
        """检索 → source_type 过滤 → 格式化 chunks → LLM 抽取 → ProfileItem + Evidence。"""
        from app.rag import retrieve

        # 多取一倍：source_type 过滤后仍有 top_k 可用
        result = retrieve(retrieval_query, company=self._company, top_k=self._top_k * 2)
        chunks = self._filter_chunks(result.chunks)
        if not chunks:
            return []

        context, ref_map = self._format_chunks(result.chunks)
        prompt = _FIELD_PROMPT.format(
            company=self._company,
            field=field_name,
            context=context,
            max_items=_MAX_ITEMS_PER_FIELD,
            max_quote=_MAX_QUOTE_CHARS,
        )

        # 重试：deepseek-v4-flash 是推理模型，偶尔返回空 content（推理耗尽预算）。
        # 只有"API 返回空串"才重试；返回 [] 表示 LLM 确实没找到 → 不重试。
        raw = ""
        for attempt in range(_MAX_LLM_RETRIES):
            try:
                raw = self._call(prompt) or ""
            except Exception:  # noqa: BLE001 网络/API 异常 → 重试
                raw = ""
            if raw.strip():
                break
            time.sleep(1.0 + attempt)  # 退避
        return self._parse_field_output(raw, ref_map)

    # ── 工具 ────────────────────────────────────────────────────
    def _filter_chunks(self, chunks: list) -> list:
        """source_type 过滤 + 截取 top_k。

        空 self._source_type = 不过滤（向后兼容）；过滤发生在 extractor 层，
        不侵入检索管线（retriever/vector_store 零改动）。
        """
        if not self._source_type:
            return chunks[: self._top_k]
        filtered = [c for c in chunks if (c.metadata or {}).get("source_type") == self._source_type]
        return filtered[: self._top_k]

    def _format_chunks(self, chunks) -> tuple[str, dict[int, dict]]:
        """格式化 chunks 为 LLM 输入，返回 (context, ref_map)。

        ref_map: {chunk_idx: {source, chapter, section, page, chunk_id}} ——
        LLM 只看到索引号，source/page 由这里补全（不可伪造）。
        """
        lines: list[str] = []
        ref_map: dict[int, dict] = {}
        for i, c in enumerate(chunks):
            meta = c.metadata or {}
            ref_map[i] = {
                "source": c.source or meta.get("source", ""),
                "source_type": meta.get("source_type", "") or "",
                "chapter": meta.get("chapter", "") or "",
                "section": meta.get("section", "") or "",
                "page": c.page,
                "chunk_id": c.chunk_id,
            }
            loc = meta.get("chapter") or meta.get("section") or "未知章节"
            page = f" 页码:{c.page}" if c.page else ""
            text = (c.text or "").strip()
            # 截短 chunk 文本：deepseek-v4-flash 是推理模型，prompt 越小推理越少，
            # content 越可能完整返回（见 _http_call 注释）
            lines.append(f"[{i}] 章节:{loc}{page}\n{text[:300]}")
            ref_map[i]["text"] = text[:100]  # 兜底 quote（LLM 漏给时用真实原文片段）
        return "\n\n".join(lines), ref_map

    def _parse_field_output(self, raw: str, ref_map: dict[int, dict]) -> list[ProfileItem]:
        """解析 LLM JSON 输出 → ProfileItem 列表（含证据归因 + 越界过滤）。"""
        items_raw = self._extract_json_array(raw)
        items: list[ProfileItem] = []
        for item in items_raw:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            desc = str(item.get("description", "")).strip()
            refs = item.get("evidence_refs", []) or []
            quotes = item.get("quotes", []) or []

            evidence: list[EvidenceRef] = []
            for idx in refs:
                ref = ref_map.get(idx)
                if ref is None:
                    continue  # 越界索引 → 静默丢弃
                # quotes 可能少于 refs（LLM 只给部分引用附引文）→ 缺失时用真实 chunk 文本兜底
                quote = str(quotes.pop(0)).strip() if quotes else ""
                if not quote:
                    quote = ref.get("text", "")
                evidence.append(
                    EvidenceRef(
                        source=ref["source"],
                        source_type=ref.get("source_type", "") or "",
                        chapter=ref["chapter"],
                        section=ref["section"],
                        page=ref["page"],
                        quote=quote[: _MAX_QUOTE_CHARS],
                        chunk_id=ref["chunk_id"],
                    )
                )
            if not evidence:
                continue  # 无有效证据的实体丢弃（抗幻觉：宁缺毋滥）
            items.append(ProfileItem(name=name, description=desc, evidence=evidence))
        return items

    @staticmethod
    def _extract_json_array(raw: str) -> list[dict]:
        """从 LLM 输出中提取 JSON 数组（容错 ```json``` 包裹 / 前后杂质）。"""
        if not raw:
            return []
        raw = raw.strip()
        if raw.startswith("```"):
            body = raw.split("```", 2)
            if len(body) >= 2:
                raw = body[1]
                if raw.lstrip().startswith("json"):
                    raw = raw.lstrip()[4:]
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            arr = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
        return [x for x in arr if isinstance(x, dict)]

    # ── LLM 调用（DeepSeek urllib）──────────────────────────────
    def _http_call(self, prompt: str) -> str:
        """DeepSeek chat completions 直连（OpenAI 兼容格式）。"""
        payload = {
            "model": settings.llm_rewrite_model,
            "messages": [
                {"role": "system", "content": "你只输出 JSON 数组，不要任何其他文字。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,  # 抽取任务低温度，保证确定性
            # deepseek-v4-flash 是推理模型：reasoning_content 先消耗 token
            #（实测单次抽取推理可占 2245+ token），max_tokens 需给 content 留足空间，
            # 否则长抽取 prompt 返回空 content
            "max_tokens": 6000,
        }
        req = urllib.request.Request(
            f"{settings.llm_rewrite_base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.deepseek_api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


# ── 画像构建入口 ───────────────────────────────────────────────
def build_profile(company: str, extractor: ProfileExtractor | None = None) -> CompanyProfile:
    """构建企业知识画像：逐字段提取 → 组装 CompanyProfile。

    Args:
        company:  目标公司（如 "小米"）。
        extractor: 可注入自定义 extractor（测试用）。

    Returns:
        CompanyProfile（含每字段 Evidence 链）。
    """
    ex = extractor or ProfileExtractor(company)
    return CompanyProfile(
        company_name=company,
        industry=_extract_industry(ex, company),
        business_segments=ex.extract_field(
            "business_segments", _FIELD_QUERIES["business_segments"].format(company=company)
        ),
        products=ex.extract_field(
            "products", _FIELD_QUERIES["products"].format(company=company)
        ),
        technologies=ex.extract_field(
            "technologies", _FIELD_QUERIES["technologies"].format(company=company)
        ),
        customers=ex.extract_field(
            "customers", _FIELD_QUERIES["customers"].format(company=company)
        ),
        geographic_markets=ex.extract_field(
            "geographic_markets", _FIELD_QUERIES["geographic_markets"].format(company=company)
        ),
        competitive_advantages=ex.extract_field(
            "competitive_advantages", _FIELD_QUERIES["competitive_advantages"].format(company=company)
        ),
        risks=ex.extract_field(
            "risks", _FIELD_QUERIES["risks"].format(company=company)
        ),
        strategic_direction=ex.extract_field(
            "strategic_direction", _FIELD_QUERIES["strategic_direction"].format(company=company)
        ),
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )


def _extract_industry(ex: ProfileExtractor, company: str) -> str:
    """行业：从"主营业务"提取结果的描述中取第一段，失败返回空串。"""
    segments = ex.extract_field(
        "industry", f"{company} 所属行业 行业定位"
    )
    return segments[0].description if segments else ""
