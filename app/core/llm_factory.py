"""
LLM Factory：统一大模型工厂。

════════════════════ 为什么需要这一层？ ════════════════════════
如果每个 Agent 都自己调 init_chat_model，会出现一堆问题：
  1. 配置重复：每个地方写一遍 model_provider/base_url/api_key，改一处漏一处
  2. 无法复用：同一个"deepseek-chat"被反复创建，浪费内存
  3. 难以管控：想给所有模型关掉代理、加超时，得改无数地方

LLM Factory 把这些收口到一处：
  - 所有 Agent 通过 get_llm("sentiment") 拿模型
  - 工厂负责路由（agent_type → 模型名）、缓存、统一配置
  - 一条硬规矩：禁止 Agent 代码里直接调 init_chat_model，一律走工厂

════════════════════ 两个核心入口 ════════════════════════
  get_llm("sentiment")              → 普通聊天模型（自由文本）
  get_structured_llm("risk", Schema) → 绑定了 Pydantic Schema 的模型（结构化输出）

参照 EduAgent backend/core/llm_factory.py 模式，适配金融投研场景。
═══════════════════════════════════════════════════════════════
"""

from typing import Any, ClassVar

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

# ════════════════════════════════════════════════════════════
# 自定义 httpx 客户端——绕过系统代理
# ════════════════════════════════════════════════════════════

# 背景：Windows 系统代理或 HTTPS_PROXY 环境变量会被 httpx 默认探测到（trust_env=True），
#       导致 DeepSeek 请求走代理后 TLS 握手失败。DeepSeek 国内可直连，不需要代理。
# 解决：创建 trust_env=False 的 httpx 客户端，完全忽略系统代理和环境变量。

_HTTP_ASYNC_CLIENT = httpx.AsyncClient(
    trust_env=False,                               # 忽略系统代理 / HTTPS_PROXY 环境变量
    timeout=httpx.Timeout(120.0, connect=15.0),    # 总超时 120 秒，建立连接超时 15 秒
)

_HTTP_SYNC_CLIENT = httpx.Client(
    trust_env=False,
    timeout=httpx.Timeout(120.0, connect=15.0),
)

# ════════════════════════════════════════════════════════════
# Agent 类型 → 模型标识符 路由表
# ════════════════════════════════════════════════════════════

# 当前所有 Agent 都用 deepseek-chat（DeepSeek 已把 coder 能力合并进 chat）。
# 保留路由表的意义在于扩展性——将来 Research Agent 想用更强的推理模型，
# 只改这一张表，不碰任何业务代码。

_AGENT_MODEL_ROUTING: dict[str, str] = {
    "sentiment":  "deepseek-chat",    # 舆情分析：新闻情感评分、热点聚类
    "risk":       "deepseek-chat",    # 风险评估：三维度综合判定
    "research":   "deepseek-chat",    # 知识检索：RAG 问答（需要推理能力）
    "financial":  "deepseek-chat",    # 财务分析：指标计算 + 业务归因
    "report":     "deepseek-chat",    # 研报生成：长文本合成
    "manager":    "deepseek-chat",    # 意图路由：用户意图分类
    "intent":     "deepseek-chat",    # 意图识别（temperature=0 求稳定）
}

# 模型标识符 → API 实际 model 名（方便切换不同模型供应商）
_MODEL_ID_MAP: dict[str, str] = {
    "deepseek-chat": "deepseek-chat",
}


class LLMFactory:
    """
    大模型工厂——全项目获取模型的唯一入口。

    ══════════════════ 设计要点 ══════════════════
    - 所有方法都是 @classmethod：不需要创建工厂实例，直接用类名调用
    - _instances 是类变量：缓存所有创建过的模型，相同参数只创建一次
    - max_retries=0：模型层不重试。重试统一由 retry.py 的 @with_retry 管理。
      不能让两个重试系统叠加——模型重 3 次 × 装饰器重 3 次 = 9 次，指数爆炸。
    - temperature=0：默认零温度——金融分析要稳定输出，不要创造性胡编
    ════════════════════════════════════════════════

    用法：
        llm = LLMFactory.get_llm("sentiment")               # 普通模型
        structured = LLMFactory.get_structured_llm(           # 结构化输出
            "risk", RiskAssessment
        )
    """

    _instances: ClassVar[dict[str, BaseChatModel]] = {}   # 类变量：模型实例缓存（缓存键 → 模型）

    # ── 内部方法 ──────────────────────────────────────────

    @classmethod
    def _get_settings(cls):
        """取配置对象（延迟导入，避免循环依赖）"""
        from app.core.config import get_settings
        return get_settings()

    @classmethod
    def _build_model_kwargs(cls, model_key: str) -> dict[str, Any]:
        """
        组装 init_chat_model 需要的所有参数。

        DeepSeek 走 OpenAI 兼容接口，所以 model_provider="openai"。
        max_retries=0——不在这里重试，让 retry.py 统一管理。
        """
        settings = cls._get_settings()
        model_id = _MODEL_ID_MAP[model_key]
        return {
            "model": model_id,
            "model_provider": "openai",                    # DeepSeek 兼容 OpenAI 接口
            "temperature": 0,                               # 默认 0：评分/分析要稳定
            "api_key": settings.deepseek_api_key,
            "base_url": settings.deepseek_base_url,
            "max_retries": 0,                               # 模型层不重试，交给 retry.py
            "http_async_client": _HTTP_ASYNC_CLIENT,
            "http_client": _HTTP_SYNC_CLIENT,
        }

    # ── 对外入口 ──────────────────────────────────────────

    @classmethod
    def get_llm(
        cls,
        agent_type: str,
        temperature: float = 0,
        streaming: bool = False,
    ) -> BaseChatModel:
        """
        获取普通聊天模型（带缓存）。

        缓存键 = 模型标识符 + temperature + streaming。
        相同参数的模型只创建一次，之后直接复用。

        Args:
            agent_type: 业务类型（sentiment / risk / research / ...）
            temperature: 温度值（0=确定性，1=创造性）
            streaming:   是否启用流式输出
        """
        if agent_type not in _AGENT_MODEL_ROUTING:
            raise ValueError(
                f"未知 Agent 类型: {agent_type}，"
                f"可选: {list(_AGENT_MODEL_ROUTING)}"
            )

        model_key = _AGENT_MODEL_ROUTING[agent_type]
        cache_key = f"{model_key}_{temperature}_{streaming}"

        if cache_key not in cls._instances:
            kwargs = cls._build_model_kwargs(model_key)
            kwargs["temperature"] = temperature
            cls._instances[cache_key] = init_chat_model(**kwargs)

        return cls._instances[cache_key]

    @classmethod
    def get_structured_llm(
        cls,
        agent_type: str,
        output_schema: type[BaseModel],
        temperature: float = 0,
    ) -> Runnable:
        """
        获取绑定了结构化输出的模型。

        调用后 LLM 直接返回 Pydantic 对象，不需要解析自由文本。
        method="function_calling" 是 DeepSeek 的强制要求——
        DeepSeek 不支持 json_schema 模式，只能用 Function Calling。

        Args:
            agent_type:    业务类型
            output_schema: Pydantic 模型类（如 RiskAssessment）
            temperature:   温度值

        Returns:
            绑定了 with_structured_output 的 Runnable
        """
        llm = cls.get_llm(agent_type, temperature=temperature)
        return llm.with_structured_output(output_schema, method="function_calling")

    @classmethod
    def clear_cache(cls) -> None:
        """清空模型实例缓存（测试 / 应用关闭时用）"""
        cls._instances.clear()


# ── 模块级便捷函数（不用写 LLMFactory.get_llm，直接 get_llm）──


def get_llm(agent_type: str, **kwargs) -> BaseChatModel:
    """
    便捷函数：获取普通聊天模型。

    等价于 LLMFactory.get_llm(agent_type, **kwargs)。
    全项目统一入口——所有 Agent 都通过这一行拿模型。
    """
    return LLMFactory.get_llm(agent_type, **kwargs)


def get_structured_llm(
    agent_type: str,
    output_schema: type[BaseModel],
    **kwargs,
) -> Runnable:
    """
    便捷函数：获取结构化输出模型。
    """
    return LLMFactory.get_structured_llm(agent_type, output_schema, **kwargs)
