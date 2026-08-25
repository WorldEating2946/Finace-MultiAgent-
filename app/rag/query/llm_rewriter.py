"""LLM 查询改写器：理解用户意图 → 生成年报术语检索变体。

规则版（RuleBasedQueryRewriter）能覆盖已入库的词汇鸿沟（同义词表），
但无法处理"未来展望"这类章节映射 / 口语意图理解 —— 这正是 LLM 改写的价值。

实现要点：
    - DeepSeek（OpenAI 兼容 API）直连，urllib 请求（零新增依赖，沿用 Gitee API 经验）；
    - 结构化输出：LLM 返回 JSON 数组，解析容错（支持 ```json``` 包裹 / 数组切片）；
    - 优雅降级：无 API key / 网络失败 / 解析失败 → 回退 RuleBasedQueryRewriter；
    - 原始 query 始终保留在最前。
"""

from __future__ import annotations

import json
import urllib.request

from app.core.config import settings
from app.rag.query.rewriter import QueryRewriter, RuleBasedQueryRewriter

_LLM_SYSTEM = "你只输出 JSON 字符串数组，不要任何其他文字。"
_LLM_PROMPT = """你是中国上市公司年报 RAG 检索的查询改写专家。
用户查询：{query}

任务：改写为 {n} 条更适合年报全文检索的查询变体。要求：
1. 保留用户原始意图；
2. 使用年报正式用语（如口语"赚了多少"→"营业收入/净利润"，"汽车"→"智能电动汽车/新能源汽车"）；
3. 各变体侧重点不同：一条偏财务术语、一条偏业务板块视角、一条尽量贴近年报章节名；
4. 变体措辞要能被年报正文中的真实写法命中。

只输出 JSON 数组，例如 ["变体1", "变体2", "变体3"]。"""


class LLMQueryRewriter(QueryRewriter):
    """基于 LLM 的查询改写；失败时回退规则版，保证检索可用性。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_queries: int | None = None,
        fallback: QueryRewriter | None = None,
        _call: callable | None = None,
    ) -> None:
        """Args:
            api_key:     DeepSeek API key（默认读 settings.deepseek_api_key）。
            base_url:    OpenAI 兼容端点（默认 https://api.deepseek.com）。
            model:       LLM 模型名（默认 deepseek-chat）。
            temperature: 采样温度（默认 settings.llm_rewrite_temperature）。
            max_queries: 生成的额外变体数（不含原始 query）。
            fallback:    LLM 失败时的回退改写器（默认 RuleBasedQueryRewriter）。
            _call:       测试 seam —— 注入 LLM 调用函数（(prompt) -> content）。
        """
        self._api_key = api_key if api_key is not None else settings.deepseek_api_key
        self._base_url = (base_url or settings.llm_rewrite_base_url).rstrip("/")
        self._model = model or settings.llm_rewrite_model
        self._temperature = (
            temperature if temperature is not None else settings.llm_rewrite_temperature
        )
        self._max = max_queries if max_queries is not None else settings.llm_rewrite_max_queries
        self._fallback = fallback or RuleBasedQueryRewriter()
        self._call = _call or self._http_call

    # ── QueryRewriter 接口 ──────────────────────────────────────
    def rewrite(self, query: str) -> list[str]:
        if not self._api_key:
            return self._fallback.rewrite(query)  # 无 key → 规则版
        try:
            content = self._call(self._build_prompt(query))
            variants = self._parse_variants(content)
        except Exception:  # noqa: BLE001 网络/解析失败 → 回退，保证检索可用
            return self._fallback.rewrite(query)
        if not variants:
            return self._fallback.rewrite(query)

        result = [query]
        for v in variants:
            v = v.strip()
            if v and v != query and v not in result:
                result.append(v)
            if len(result) > self._max:
                break
        return result if len(result) > 1 else self._fallback.rewrite(query)

    # ── Prompt / LLM 调用 ───────────────────────────────────────
    def _build_prompt(self, query: str) -> str:
        return _LLM_PROMPT.format(query=query, n=self._max)

    def _http_call(self, prompt: str) -> str:
        """DeepSeek chat completions 直连（OpenAI 兼容格式）。"""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": 300,
        }
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_variants(content: str) -> list[str]:
        """从 LLM 输出中提取 JSON 字符串数组（容错 ```json``` 包裹 / 前后杂质）。"""
        if not content:
            return []
        content = content.strip()
        if content.startswith("```"):
            body = content.split("```", 2)
            if len(body) >= 2:
                content = body[1]
                if content.lstrip().startswith("json"):
                    content = content.lstrip()[4:]
        start, end = content.find("["), content.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            arr = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return []
        return [str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()]
