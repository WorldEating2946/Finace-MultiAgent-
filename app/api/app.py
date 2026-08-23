"""FastAPI 应用工厂（PR40 + PR41 lifespan）。

    from app.api.app import create_app
    app = create_app()
    # uvicorn app.api.app:create_app 或 python -c "..." 启动

统一异常处理：AppError → JSONResponse {code, message, data}。
生命周期（PR41）：shutdown 时关闭 ResearchService（TaskManager worker + 连接池）。
测试：app.dependency_overrides[get_research_service] = lambda: mock_service
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.exceptions import AppError
from app.core.response import error


def create_app() -> FastAPI:
    """构建 FinaceAgent Research Service FastAPI 应用。"""

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        """应用生命周期：启动时挂载单例 service，关闭时释放资源。"""
        from app.api import research as research_api
        from app.rag.vectorstore.health import check_backend_ready

        # PR44.4 向量后端启动检查：milvus 未就绪 → fail fast（无自动 fallback）
        problems = check_backend_ready()
        if problems:
            raise RuntimeError("向量后端启动检查失败: " + " | ".join(problems))

        service = research_api.get_research_service()
        app.state.research_service = service
        try:
            yield
        finally:
            # 关闭后台 worker + 业务/checkpoint 连接池（优雅下线）
            await service.shutdown()

    app = FastAPI(
        title="FinaceAgent Research Service",
        description="企业研究智能体服务：创建 / 观察 / 干预 / 恢复研究任务",
        version="0.1.0",
        lifespan=_lifespan,
    )

    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        """业务异常 → 统一响应信封。"""
        return JSONResponse(status_code=exc.http_status, content=error(exc))

    app.include_router(api_router)
    return app
