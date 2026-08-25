"""研究报告生成（PR #37）。

结构化报告：title + summary + advantages/risks/uncertainties，每条 claim 带证据链。

抗幻觉设计（同 ProfileExtractor，PR #34 模式）：
    - LLM 只返回证据索引号（evidence_refs），真实 EvidenceRef 由后处理从 ref_map 补全；
    - 越界索引被静默丢弃；
    - 无有效证据的 claim 保留但 evidence 为空（宁缺毋滥）。

LLM 模式：DeepSeek（OpenAI 兼容），urllib 直连（复用 llm_rewriter/profile 模式）。
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.core.config import settings
from app.rag.profile.schema import EvidenceRef
from app.rag.research.state import ResearchState

_MAX_CLAIMS = 6          # advantages / risks 各最多条数
_MAX_UNCERTAINTIES = 4   # uncertainties 最多条数
_MAX_LLM_RETRIES = 3     # 空 content 重试（deepseek-v4-flash 推理模型间歇性返回空）

_REPORT_PROMPT = """你是金融研究分析专家。根据以下企业研究计划与收集的证据，撰写结构化研究报告。

研究问题：{request}
研究意图：{intent}
公司：{company}
研究步骤：
{plan_summary}

证据资料（每条带 [索引] 和来源）：
{context}

要求：
1. 只基于证据资料撰写，不编造、不推断；证据未覆盖的内容不写；
2. advantages: 该企业的竞争优势（每条 claim 一句话，evidence_refs 为支撑该论点的证据索引）；
3. risks: 该企业面临的风险挑战；
4. uncertainties: 证据不足或存在矛盾的事项（一句话描述）；
5. evidence_refs 中的索引必须真实存在于证据资料中，否则该 claim 丢弃。

只输出 JSON，不要任何其他文字，格式：
{{"title":"...","summary":"一句话总结","advantages":[{{"claim":"...","evidence_refs":[0,1]}}],"risks":[{{"claim":"...","evidence_refs":[2]}}],"uncertainties":["..."]}}"""


class ReportClaim(BaseModel):
    """报告中的单条论点——claim + 支撑证据。"""

    claim: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """结构化研究报告（含完整证据链，可审计）。"""

    title: str
    summary: str
    plan_summary: list[str] = Field(default_factory=list)   # 研究步骤摘要
    advantages: list[ReportClaim] = Field(default_factory=list)
    risks: list[ReportClaim] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)  # 完整证据链
    generated_at: str = ""


class ReportBuilder:
    """报告生成器：evidence_pool → LLM 合成 → 结构化 ResearchReport。"""

    def __init__(self, *, _call: callable | None = None) -> None:
        """Args:
            _call: 测试 seam —— 注入 LLM 调用函数（prompt -> content）。
        """
        self._call = _call or self._http_call

    # ── 主入口 ─────────────────────────────────────────────────
    def build(self, state: ResearchState) -> ResearchReport:
        """格式化证据 → LLM 合成 → 解析 evidence_refs → ResearchReport。"""
        context, ref_map = self._format_evidence(state)
        prompt = _REPORT_PROMPT.format(
            request=state.request,
            intent=state.intent,
            company=state.target.company,
            plan_summary="\n".join(f"{s.order}. {s.name}" for s in state.plan.steps),
            context=context,
        )
        raw = ""
        for attempt in range(_MAX_LLM_RETRIES):
            try:
                raw = self._call(prompt) or ""
            except Exception:  # noqa: BLE001 网络/API 异常 → 重试
                raw = ""
            if raw.strip():
                break
            time.sleep(1.0 + attempt)
        return self._parse_report(raw, ref_map)

    # ── 证据格式化（LLM 只见索引，source 由 ref_map 补全）────────
    @staticmethod
    def _format_evidence(state: ResearchState) -> tuple[str, dict[int, EvidenceRef]]:
        """证据格式化为 LLM 输入，返回 (context, ref_map)。

        ref_map: {idx: EvidenceRef} —— LLM 只返回索引，source/page/quote 不可伪造。
        """
        lines: list[str] = []
        ref_map: dict[int, EvidenceRef] = {}
        # 快速模式 max_evidence 压缩证据上下文，缩短 LLM 报告合成长
        pool = state.evidence_pool[: getattr(state, "max_evidence", 60)]
        for i, e in enumerate(pool):
            ref_map[i] = e
            loc = e.chapter or e.section or "未知章节"
            page = f" 页码:{e.page}" if e.page else ""
            lines.append(
                f"[{i}] 来源:{e.source_type or '年报'}{page}\n{e.quote}"
            )
        return "\n\n".join(lines), ref_map

    # ── 报告解析（抗幻觉）──────────────────────────────────────
    def _parse_report(self, raw: str, ref_map: dict[int, EvidenceRef]) -> ResearchReport:
        """解析 LLM JSON → ResearchReport（越界 evidence_refs 静默丢弃）。"""
        data = self._extract_json_object(raw)
        advantages = self._parse_claims(data.get("advantages"), ref_map)
        risks = self._parse_claims(data.get("risks"), ref_map)
        uncertainties = [
            str(u).strip() for u in (data.get("uncertainties") or [])
            if str(u).strip()
        ][:_MAX_UNCERTAINTIES]
        # 收集被引用的证据（去重保序）
        evidence = self._collect_evidence(advantages, risks, ref_map)
        return ResearchReport(
            title=str(data.get("title", "")).strip(),
            summary=str(data.get("summary", "")).strip(),
            advantages=advantages,
            risks=risks,
            uncertainties=uncertainties,
            evidence=evidence,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _parse_claims(claims_raw, ref_map: dict[int, EvidenceRef]) -> list[ReportClaim]:
        """解析 claims：claim + 越界过滤的 evidence_refs。"""
        claims: list[ReportClaim] = []
        for item in claims_raw or []:
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            refs = item.get("evidence_refs", []) or []
            evidence = [ref_map[i] for i in refs if isinstance(i, int) and i in ref_map]
            claims.append(
                ReportClaim(
                    claim=claim,
                    evidence=evidence,
                    source_types=list({e.source_type for e in evidence if e.source_type}),
                )
            )
            if len(claims) >= _MAX_CLAIMS:
                break
        return claims

    @staticmethod
    def _collect_evidence(advantages, risks, ref_map: dict[int, EvidenceRef]) -> list[EvidenceRef]:
        """收集报告引用的证据（去重保序，供审计）。"""
        seen: set[str] = set()
        result: list[EvidenceRef] = []
        for claim in list(advantages) + list(risks):
            for e in claim.evidence:
                if e.chunk_id and e.chunk_id in seen:
                    continue
                if e.chunk_id:
                    seen.add(e.chunk_id)
                result.append(e)
        return result

    @staticmethod
    def _extract_json_object(raw: str) -> dict:
        """从 LLM 输出提取 JSON 对象（容错 ```json``` 包裹 / 前后杂质）。"""
        if not raw:
            return {}
        raw = raw.strip()
        if raw.startswith("```"):
            body = raw.split("```", 2)
            if len(body) >= 2:
                raw = body[1]
                if raw.lstrip().startswith("json"):
                    raw = raw.lstrip()[4:]
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            obj = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return {}
        return obj if isinstance(obj, dict) else {}

    # ── LLM 调用（DeepSeek urllib）──────────────────────────────
    def _http_call(self, prompt: str) -> str:
        """DeepSeek chat completions 直连（OpenAI 兼容格式）。

        deepseek-v4-flash 是推理模型：reasoning_content 先耗 token，
        max_tokens=6000 给 content 留足空间（同 ProfileExtractor 经验）。
        """
        payload = {
            "model": settings.llm_rewrite_model,
            "messages": [
                {"role": "system", "content": "你只输出 JSON 对象，不要任何其他文字。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,  # 报告生成低温度，保证确定性
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
