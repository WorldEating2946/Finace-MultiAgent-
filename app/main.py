"""
app/main.py — FastAPI 服务入口（统一入口：Financial Agent 全链路 + Sentiment & Risk API）

⚠️ 多入口并存说明:
    - 本入口 `app.main:app` = Financial Agent 独立工作流（POST /analyze 全链路）
      + Sentiment & Risk Agent API（POST /api/v1/sentiment | /api/v1/risk |
      /api/v1/sentiment-risk/full）。默认端口与 app/api/app.py 相同（8000），两者**勿同时启动**；
      如需并行，请用 `APP_PORT` 环境变量给其中一个分配不同端口。
    - Research Agent 生产 API 在 `app/api/app.py`（POST /research/start + SSE 流式，
      含 lifespan 向量后端健康检查 fail-fast）——生产主入口。

启动:
    conda activate finance-agent
    set FINANCE_AGENT_DATA_FILE=%cd%\app\\data\fixtures\\sample_company.json
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

API:
    POST /analyze            — 投研分析（Financial 全链路）
    POST /api/v1/sentiment   — 舆情分析（Sentiment Agent）
    POST /api/v1/risk        — 风险评估（Sentiment + Risk）
    POST /api/v1/sentiment-risk/full — 完整链路联合输出
    POST /api/v1/report/generate — 统一研报生成（四 Agent 输出 → Markdown+HTML）
    POST /api/v1/analyze/stream  — 投研分析 SSE 全链路可视化流
    GET  /health             — 健康检查

Author: 工藤 (Financial Agent), 成员 (Sentiment & Risk Agent), 合并整理: Claude
Date: 2026-08-05
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from app.api.analyze_stream import router as analyze_stream_router
from app.api.auth import router as auth_router
from app.api.financial import router as financial_router
from app.api.history import router as history_router
from app.api.knowledge import router as knowledge_router
from app.api.report import router as report_router
from app.api.research_entry import router as research_entry_router
from app.api.routes import router as sentiment_risk_router
from app.api.system_status import router as system_status_router
from app.core.logging_config import get_logger, setup_logging
from app.database.session import init_db
from app.models.user import User
from app.services.auth import get_current_user
from app.workflow.graph import build_graph

# 加载 .env + 初始化日志
load_dotenv()
setup_logging()

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """应用生命周期：启动时建表（users 等新增表，幂等）。

    DB 不可达仅告警不崩溃——/health 仍可用；受鉴权保护的端点
    （/analyze、/api/v1/analyze/stream、/api/v1/report/generate）会因 DB 依赖返回
    明确错误。
    """
    try:
        await init_db()
    except Exception:  # noqa: BLE001 —— 启动不强依赖 DB，便于本地快速调试
        logger.warning("[lifespan] 数据库建表失败（可能 DB 未就绪），鉴权端点将不可用", exc_info=True)
    yield


app = FastAPI(
    title="FinaceAgent API",
    description="多 Agent 金融智能分析平台",
    version="0.3.0",
    lifespan=_lifespan,
)

# CORS — 允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sentiment & Risk Agent 路由（/api/v1 前缀）
app.include_router(sentiment_risk_router)

# 认证路由（/api/v1/auth 前缀；login/refresh 开放，me/logout/users 受保护）
app.include_router(auth_router)

# 各 Agent 独立入口（单 Agent 可独立运作，无需跑整条主链）
app.include_router(financial_router)
app.include_router(research_entry_router)
app.include_router(knowledge_router, prefix="/api/v1")  # /api/v1/knowledge/search + /upload
app.include_router(history_router)  # /api/v1/history 保存/列出 Agent 运行历史
app.include_router(system_status_router)  # /api/v1/health/status 系统就绪状态

# 统一研报生成 + SSE 全链路可视化路由（成员 5：平台工程与报告生成）
app.include_router(report_router)
app.include_router(analyze_stream_router)

# 全局图实例（模块级单例，启动时编译一次）
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        logger.info("编译 LangGraph 工作流...")
        _graph = build_graph()
    return _graph


# ============================================================================
# 请求/响应模型
# ============================================================================


class AnalyzeRequest(BaseModel):
    company: str = Field(..., min_length=1, description="目标公司名称，如 '宁德时代'")
    ticker: str = Field(default="", description="股票代码（可选），如 '300750'")
    user_query: str = Field(default="", description="用户提问（可选）")

    model_config = ConfigDict(extra="forbid")


class AnalyzeResponse(BaseModel):
    company: str
    ticker: str
    report: str
    started_at: str = ""
    completed_at: str = ""
    errors: list[dict] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

    model_config = ConfigDict(extra="forbid")


# ============================================================================
# 路由
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return HealthResponse(
        status="ok",
        version="0.3.0",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, _: User = Depends(get_current_user)):
    """执行完整投研分析流程。

    输入目标公司 → Manager 规划 → [Research ‖ Financial ‖ Sentiment] → Risk → Report
    """
    logger.info("收到分析请求: company=%s, ticker=%s", req.company, req.ticker)

    graph = get_graph()

    initial_state = {
        "company": req.company,
        "ticker": req.ticker,
        "user_query": req.user_query or f"分析{req.company}的财务健康状况与发展前景",
        "current_step": "start",
        "errors": [],
    }

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("工作流执行失败")
        raise HTTPException(status_code=500, detail=f"分析失败: {exc}")

    return AnalyzeResponse(
        company=final_state.get("company", req.company),
        ticker=final_state.get("ticker", req.ticker),
        report=final_state.get("report", "（报告生成失败）"),
        started_at=final_state.get("started_at", ""),
        completed_at=final_state.get("completed_at", ""),
        errors=final_state.get("errors", []),
    )


# ============================================================================
# 启动入口
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
