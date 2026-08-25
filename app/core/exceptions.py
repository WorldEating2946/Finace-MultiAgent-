"""统一业务异常（PR40）。

AppError 体系：所有业务异常携带业务错误码（code）与 HTTP 状态码，
由 FastAPI exception_handler（app/api/app.py）转成统一响应信封 {code, message, data}。
"""

from __future__ import annotations


class AppError(Exception):
    """业务异常基类。"""

    code: int = 10000          # 业务错误码
    message: str = "error"     # 人类可读信息
    http_status: int = 400     # HTTP 状态码

    def __init__(self, message: str | None = None) -> None:
        if message:
            self.message = message
        super().__init__(self.message)


class ResearchNotFound(AppError):
    """研究任务不存在。"""

    code = 40001
    message = "research task not found"
    http_status = 404


class CheckpointExpired(AppError):
    """任务 checkpoint 已失效（如持久化被清理）。"""

    code = 40002
    message = "research checkpoint expired"
    http_status = 410


class InvalidDecision(AppError):
    """人工审核决策非法（action 不在 approve/reject/modify）。"""

    code = 40003
    message = "invalid human decision"
    http_status = 400


class ResearchReportNotReady(AppError):
    """报告未生成（任务未完成 / 未到报告节点）。"""

    code = 40004
    message = "research report not ready"
    http_status = 409
