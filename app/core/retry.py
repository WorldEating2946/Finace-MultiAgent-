"""
三层兜底机制：自动重试 → Agent 级降级 → 系统级兜底。

════════════════════ 为什么需要三层？ ════════════════════════
线上 LLM 超时、网络抖动、API 限流 —— 不能一出错就让用户看到崩溃界面。
三层兜底保证：无论发生什么，用户始终能收到有意义的响应。

层级   | 触发条件                   | 处理方式
Layer1 | 网络抖动 / LLM 超时等短暂故障 | 自动重试最多 2 次（间隔 1s / 3s）
Layer2 | 重试 2 次仍失败             | Agent 级降级（静态提示 / 标记需人工复核）
Layer3 | 连降级都失败                | 系统级兜底（友好提示，永远不崩）

════════════════════ 装饰器设计 ════════════════════════
with_retry 是一个「装饰器工厂」——三层嵌套函数：

  def with_retry(agent_type=""):      # 最外层：接收参数（agent_type），返回装饰器
      def decorator(func):            # 中间层：接收被装饰的函数
          @wraps(func)
          async def wrapper(...):     # 最内层：真正的执行逻辑（三层兜底都在这里）
              ...

为什么三层嵌套？因为我们需要给装饰器传参数（agent_type）。
如果不传参数，两层就够：def retry(func) → async def wrapper(...)。

════════════════════ 关键设计决策 ════════════════════════
1. 不可重试异常立即抛出：用户输入非法 / 认证失败 → 重试一万次也没用
2. 降级函数返回 dict：因为符合 LangGraph 节点约定（返回要更新的 State 字段）
3. 只有一处真实调用：@with_retry 在 orchestrator 里只装饰 _invoke() 闭包，
   args=() 始终为空，降级函数不依赖外部参数
═══════════════════════════════════════════════════════════
"""

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any

# ════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════

MAX_RETRIES = 2                    # 最多重试次数（总共 3 次尝试：初次 + 2 次重试）
RETRY_DELAYS = [1.0, 3.0]         # 递增等待间隔（1 秒 → 3 秒）
SINGLE_CALL_TIMEOUT = 30.0         # 单次调用超时（秒），防止 LLM 卡死

# ── 异常分类 ──────────────────────────────────────────────
# 可重试：网络问题、超时等短暂故障，重试可能恢复
RETRYABLE_ERRORS = (TimeoutError, ConnectionError, OSError, RuntimeError)

# 不可重试：逻辑错误，重试没有意义，应立即抛出
NON_RETRYABLE_ERRORS = (ValueError, TypeError, KeyError)


# ════════════════════════════════════════════════════════════
# 降级函数 —— Agent 级备选方案
# ════════════════════════════════════════════════════════════

class AgentFallbackHandler:
    """
    第二层兜底：Agent 级降级。

    每个降级函数返回一个 dict——符合 LangGraph 节点约定
    （节点函数接收 State，返回要更新的字段 dict）。

    降级策略：
      - 舆情/风险/知识/财务 → 返回静态提示 + fallback_used=True
      - 试卷批改（EduAgent 模式，FinaceAgent 暂未用到） → 标记需教师复核
    """

    @classmethod
    async def handle(cls, agent_type: str, original_error: Exception | None = None) -> dict:
        """根据 agent_type 调用对应降级策略"""
        strategy = {
            "sentiment":  cls._sentiment_fallback,
            "risk":       cls._risk_fallback,
            "research":   cls._research_fallback,
            "financial":  cls._financial_fallback,
            "report":     cls._report_fallback,
            "manager":    cls._manager_fallback,
        }.get(agent_type, cls._default_fallback)
        return strategy(original_error)

    @classmethod
    def _sentiment_fallback(cls, error) -> dict:
        return {"fallback_used": True, "summary": "舆情分析服务暂时不可用，请稍后重试。"}

    @classmethod
    def _risk_fallback(cls, error) -> dict:
        return {"fallback_used": True, "risk_summary": "风险评估服务暂时不可用，请稍后重试。"}

    @classmethod
    def _research_fallback(cls, error) -> dict:
        return {"fallback_used": True, "content": "知识检索服务暂时不可用。"}

    @classmethod
    def _financial_fallback(cls, error) -> dict:
        return {"fallback_used": True, "metrics": {}, "summary": "财务数据获取失败，请稍后重试。"}

    @classmethod
    def _report_fallback(cls, error) -> dict:
        return {"fallback_used": True, "content": "报告生成服务暂时不可用，请稍后重试。"}

    @classmethod
    def _manager_fallback(cls, error) -> dict:
        return {"fallback_used": True, "content": "系统编排服务暂时不可用，请稍后重试。"}

    @classmethod
    def _default_fallback(cls, error) -> dict:
        return {"fallback_used": True, "content": "服务暂时不可用，请稍后重试。"}


# ════════════════════════════════════════════════════════════
# 系统级兜底
# ════════════════════════════════════════════════════════════

def _system_fallback_response(agent_type: str) -> dict:
    """
    第三层兜底：系统级保底。

    这是最后的防线——无论前面怎么崩，这个函数永远不会失败。
    返回一段友好提示，同时标记 system_fallback=True，
    让前端可以展示"系统异常"的视觉提示。
    """
    return {
        "fallback_used": True,
        "system_fallback": True,
        "content": "很抱歉，系统遇到了暂时无法恢复的错误。请稍后重试或联系管理员。",
    }


# ════════════════════════════════════════════════════════════
# 装饰器工厂
# ════════════════════════════════════════════════════════════

def with_retry(agent_type: str = ""):
    """
    三层兜底装饰器工厂。

    用法:
        @with_retry(agent_type="sentiment")
        async def _invoke():
            return await graph.ainvoke(state, config=config)

    执行流程：
      1. 第一层：asyncio.wait_for 包装，30 秒超时，失败后最多重试 2 次
      2. 第二层：重试耗尽 → AgentFallbackHandler.handle() 降级
      3. 第三层：降级失败 → _system_fallback_response() 兜底保证有响应

    Args:
        agent_type: 业务类型标识，用于选择降级策略（sentiment/risk/research/...）
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)                         # 保留原函数的 __name__ / __doc__ 等属性
        async def wrapper(*args, **kwargs) -> Any:
            last_error = None

            # ── 第一层：带超时的自动重试 ──────────────────
            for attempt in range(MAX_RETRIES + 1):        # 最多 3 次尝试
                try:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),            # 执行原函数
                        timeout=SINGLE_CALL_TIMEOUT,      # 单次 30 秒超时
                    )
                    return result                         # 成功 → 直接返回
                except NON_RETRYABLE_ERRORS:
                    raise                                 # 不可重试 → 立即抛出
                except Exception as e:  # noqa: BLE001 —— 重试循环统一捕获，记录 last_error 供降级层使用
                    last_error = e
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_DELAYS[attempt])  # 等待后重试

            # ── 第二层：Agent 级降级 ──────────────────────
            try:
                return await AgentFallbackHandler.handle(
                    agent_type=agent_type,
                    original_error=last_error,
                )
            except Exception:  # noqa: BLE001, S110 —— 降级也失败 → 进第三层系统级兜底
                pass                           # 降级也失败 → 进第三层

            # ── 第三层：系统级兜底 ────────────────────────
            return _system_fallback_response(agent_type)

        return wrapper

    return decorator
