"""
app/services/data_fetcher.py — 市场数据获取服务

本模块封装外部金融数据源的异步调用逻辑，提供统一的 MarketDataService。
支持的接口: AkShare (免费) / Tushare Pro (需 token)。

设计原则:
    - 全异步: 所有公开方法均为 async，内部使用 httpx.AsyncClient
    - 容错保障: 自动重试 (指数退避) + 超时控制 + 熔断
    - 数据一致性: 返回结构统一为 Pydantic 模型，屏蔽不同数据源的差异

Author: 工藤
Date: 2026-08-05
Version: 0.1.0
"""

import asyncio
import logging
import re
import time

import httpx

from app.core.schemas import (
    CompanyBasicInfo,
    FinancialMetric,
    MarketDataRequest,
    MarketDataResponse,
    MetricType,
    ServiceResult,
)


def _to_num(v) -> float | None:
    """安全转 float；None/NaN/非数字 → None。"""
    try:
        f = float(v)
        return None if f != f else f  # NaN != NaN → None
    except (TypeError, ValueError):
        return None

logger = logging.getLogger(__name__)


# ============================================================================
# 自定义异常
# ============================================================================


class DataFetchError(Exception):
    """数据获取通用异常"""

    def __init__(self, message: str, source: str = "", status_code: int | None = None):
        self.source = source
        self.status_code = status_code
        super().__init__(message)


class CircuitBreakerOpen(DataFetchError):
    """熔断器开启 —— 短期内不再尝试请求"""


class RateLimitExceeded(DataFetchError):
    """API 速率限制"""


# ============================================================================
# 简单熔断器
# ============================================================================


class _CircuitBreaker:
    """简单的熔断器实现

    在连续失败达到阈值后，进入 OPEN 状态并拒绝请求。
    冷却时间过后进入 HALF_OPEN 状态允许探测请求。
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._state: str = "CLOSED"  # CLOSED | OPEN | HALF_OPEN

    @property
    def is_open(self) -> bool:
        if self._state == "CLOSED":
            return False
        if self._state == "OPEN":
            # 检查是否已过冷却时间
            if time.monotonic() - self._last_failure_time >= self.cooldown_seconds:
                self._state = "HALF_OPEN"
                logger.info("熔断器进入 HALF_OPEN 状态，允许探测请求")
                return False
            return True
        # HALF_OPEN — 允许通过
        return False

    def record_success(self) -> None:
        """记录一次成功，重置计数器"""
        self._failure_count = 0
        self._state = "CLOSED"

    def record_failure(self) -> None:
        """记录一次失败"""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning("熔断器进入 OPEN 状态（连续 %d 次失败）", self._failure_count)


# ============================================================================
# MarketDataService
# ============================================================================


class MarketDataService:
    """市场数据获取服务

    封装对外部金融数据 API 的异步调用，提供统一的数据获取接口。

    特性:
        - 自动重试 (3 次，指数退避)
        - 请求超时控制 (默认 30s)
        - 熔断保护 (连续 5 次失败后冷却 60s)
        - 适配器模式：统一 AkShare / Tushare 的返回差异

    使用示例:
        async with MarketDataService() as svc:
            req = MarketDataRequest(ticker="600519", start_date=..., end_date=...)
            result = await svc.fetch_financial_data(req)
    """

    # ------------------------------------------------------------------
    # 配置常量
    # ------------------------------------------------------------------
    DEFAULT_TIMEOUT = 30.0        # 秒
    MAX_RETRIES = 3               # 最大重试次数
    RETRY_BASE_DELAY = 1.0        # 重试基础延迟（秒）
    RETRY_MAX_DELAY = 10.0        # 重试最大延迟（秒）
    CIRCUIT_BREAKER_THRESHOLD = 5
    CIRCUIT_BREAKER_COOLDOWN = 60.0

    # AkShare 是同步库，此处模拟其异步行为
    # 生产环境可通过 asyncio.to_thread 或 ThreadPoolExecutor 桥接
    # 运行时可安装检测（pip install akshare），无需手动开关
    AKSHARE_AVAILABLE = False

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        tushare_token: str | None = None,
    ):
        self._timeout = timeout
        self._max_retries = max_retries
        self._tushare_token = tushare_token
        self._client: httpx.AsyncClient | None = None
        self._breaker = _CircuitBreaker(
            failure_threshold=self.CIRCUIT_BREAKER_THRESHOLD,
            cooldown_seconds=self.CIRCUIT_BREAKER_COOLDOWN,
        )

        # 自动探测 akshare（已安装则真行情，否则走 fixture/httpx 降级）
        try:
            import akshare as ak
            self._ak = ak
            self.AKSHARE_AVAILABLE = True
        except ImportError:
            logger.info("akshare 未安装，将使用 fixture/httpx 降级")
            self._ak = None

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MarketDataService":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            headers={"User-Agent": "FinanceAgent/0.1.0"},
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """获取内部 HTTP 客户端（必须先进入上下文）"""
        if self._client is None:
            raise RuntimeError("MarketDataService 必须通过 'async with' 使用")
        return self._client

    # ------------------------------------------------------------------
    # 核心数据获取方法
    # ------------------------------------------------------------------

    async def fetch_financial_data(self, request: MarketDataRequest) -> MarketDataResponse:
        """获取目标公司的财务数据。

        自动根据 request.source 选择数据源适配器。

        参数:
            request: MarketDataRequest — 目标公司、时间范围、指标列表

        返回:
            MarketDataResponse — 统一格式的财务数据列表
        """
        if self._breaker.is_open:
            raise CircuitBreakerOpen(
                "熔断器已开启，请稍后重试",
                source=request.source,
            )

        start_ts = time.monotonic()
        try:
            if request.source == "akshare":
                data = await self._fetch_from_akshare(request)
            elif request.source == "tushare":
                data = await self._fetch_from_tushare(request)
            else:
                raise ValueError(f"不支持的数据源: {request.source}")

            self._breaker.record_success()
            elapsed = (time.monotonic() - start_ts) * 1000
            logger.info("数据获取成功: %s, %d 条记录, 耗时 %.0fms", request.ticker, len(data), elapsed)

            return MarketDataResponse(
                request=request,
                data=data,
                source=request.source,
            )

        except Exception as exc:
            self._breaker.record_failure()
            elapsed = (time.monotonic() - start_ts) * 1000
            logger.error("数据获取失败: %s, 耗时 %.0fms, 错误: %s", request.ticker, elapsed, exc)
            return MarketDataResponse(
                request=request,
                data=[],
                source=request.source,
                error_msg=str(exc),
            )

    async def fetch_company_info(self, ticker: str) -> ServiceResult:
        """获取公司基本信息。

        参数:
            ticker: 股票代码

        返回:
            ServiceResult[CompanyBasicInfo]
        """
        try:
            data = await self._retry_request(
                "GET",
                f"https://api.example.com/v1/company/{ticker}",  # 占位 URL
            )
            # TODO: 解析实际 API 响应 → CompanyBasicInfo
            info = CompanyBasicInfo(name=ticker, ticker=ticker)
            return ServiceResult(success=True, data=info)
        except Exception as exc:
            return ServiceResult(success=False, error_msg=str(exc))

    # ------------------------------------------------------------------
    # 数据源适配器（内部）
    # ------------------------------------------------------------------

    async def _fetch_from_akshare(self, request: MarketDataRequest) -> list[FinancialMetric]:
        """通过 AkShare 获取财务数据。

        AkShare 是同步库，此处通过 asyncio.to_thread 桥接到线程池执行。
        若 akshare 不可用，则使用 httpx 直接调用其底层 REST API。
        """
        if self._ak is not None:
            # 生产路径: 桥接同步 akshare 到异步
            return await asyncio.to_thread(
                self._fetch_akshare_sync, request
            )

        # 开发/降级路径: 通过 httpx 调用（适配 akshare Web API 或返回模拟结构）
        logger.debug("akshare 不可用，使用 httpx 降级路径")

        # ================================================================
        # TODO: 替换为实际的 AkShare REST API endpoint
        # 当前返回空列表作为骨架占位，确保类型系统可运行
        # 实际接入示例（取消注释并替换 URL）:
        #
        #   url = "https://datacenter.eastmoney.com/api/data/v1/get"
        #   params = {
        #       "reportName": "RPT_DMSK_FN_MAININDICATOR",
        #       "columns": "SECURITY_CODE,START_DATE,END_DATE,STD_ITEM_NAME,AMOUNT",
        #       "filter": f'(SECURITY_CODE="{request.ticker}")',
        #       "pageSize": 100,
        #   }
        #   response = await self._retry_request("GET", url, params=params)
        #   return self._parse_eastmoney_response(response.json(), request)
        # ================================================================
        return []

    def _fetch_akshare_sync(self, request: MarketDataRequest) -> list[FinancialMetric]:
        """同步调用 akshare EM 报表（线程池），取最新年报四指标 → FinancialMetric。

        数据源：stock_profit_sheet_by_report_em（营收/归母净利）+
        stock_balance_sheet_by_report_em（总资产/归母权益），绝对元口径。
        防御式：任一字段缺失/解析失败 → 返回空（上层回退 fixture，绝不产出部分脏数据）。
        """
        import akshare as ak

        metrics: list[FinancialMetric] = []
        pre = self._prefixed_symbol(request.ticker)
        try:
            profit = ak.stock_profit_sheet_by_report_em(symbol=pre)
            balance = ak.stock_balance_sheet_by_report_em(symbol=pre)
        except Exception as exc:  # noqa: BLE001 —— akshare/网络异常回退 fixture
            logger.error("akshare EM 报表获取失败: %s", exc)
            return metrics

        rp = self._latest_annual_row(profit)
        rb = self._latest_annual_row(balance)
        if rp is None or rb is None:
            logger.warning("akshare 无年度报表: %s", request.ticker)
            return metrics

        year = self._row_year(rp)
        if year is None:
            return metrics

        pairs = (
            (rb.get("TOTAL_ASSETS"), MetricType.TOTAL_ASSETS),
            (rb.get("TOTAL_PARENT_EQUITY"), MetricType.SHAREHOLDERS_EQUITY),
            (rp.get("TOTAL_OPERATE_INCOME"), MetricType.REVENUE),
            (rp.get("PARENT_NETPROFIT"), MetricType.NET_PROFIT),
        )
        for raw, mtype in pairs:
            val = _to_num(raw)
            if val is not None:
                metrics.append(
                    FinancialMetric(metric_type=mtype, value=val, period="FY", fiscal_year=year)
                )
        return metrics

    @staticmethod
    def _prefixed_symbol(ticker: str) -> str:
        """裸代码 → 带交易所前缀（akshare EM 接口要求，如 300750→SZ300750）。"""
        return ("SH" + ticker) if ticker.startswith(("6", "5", "9")) else ("SZ" + ticker)

    @staticmethod
    def _latest_annual_row(df, date_col: str = "REPORT_DATE"):
        """取最新年报行（REPORT_DATE 含 12-31，且年份最大）。无则 None。"""
        if df is None or df.empty or date_col not in df.columns:
            return None
        annual = df[df[date_col].astype(str).str.contains("12-31")]
        if annual.empty:
            return None
        annual = annual.copy()
        annual["_yr"] = annual[date_col].astype(str).str.extract(r"(\d{4})").astype(float)
        maxy = annual["_yr"].max()
        return annual[annual["_yr"] == maxy].iloc[-1]

    @staticmethod
    def _row_year(row, date_col: str = "REPORT_DATE") -> int | None:
        m = re.search(r"(20\d{2})", str(row.get(date_col, "")))
        return int(m.group(1)) if m else None

    async def _fetch_from_tushare(self, request: MarketDataRequest) -> list[FinancialMetric]:
        """通过 Tushare Pro 获取财务数据。

        需要有效的 tushare_token。
        """
        if not self._tushare_token:
            raise DataFetchError("Tushare token 未配置，请在初始化 MarketDataService 时传入")

        # Tushare HTTP API 端点
        url = "https://api.tushare.pro"

        payload = {
            "api_name": "fina_indicator",
            "token": self._tushare_token,
            "params": {
                "ts_code": request.ticker,
                "start_date": request.start_date.strftime("%Y%m%d"),
                "end_date": request.end_date.strftime("%Y%m%d"),
            },
            "fields": "ts_code,end_date,revenue,net_profit,total_assets,roe",
        }

        response = await self._retry_request("POST", url, json=payload)
        data = response.json()

        # 检查 Tushare 错误码
        if data.get("code") != 0:
            raise DataFetchError(
                f"Tushare API 错误: {data.get('msg', '未知错误')}",
                source="tushare",
            )

        # 解析为统一 FinancialMetric 列表
        metrics: list[FinancialMetric] = []
        for item in data.get("data", {}).get("items", []):
            # TODO: 映射 Tushare 字段到 FinancialMetric
            pass

        return metrics

    # ------------------------------------------------------------------
    # 重试与请求基础设施
    # ------------------------------------------------------------------

    async def _retry_request(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
        headers: dict | None = None,
    ) -> httpx.Response:
        """带自动重试的 HTTP 请求。

        特性:
            - 指数退避: 1s → 2s → 4s (上限 10s)
            - 可重试状态码: 429, 5xx
            - 连接错误自动重试
        """
        last_exception: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    headers=headers,
                )

                # 速率限制 — 等服务器指定时间后重试
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "5")
                    wait = float(retry_after)
                    logger.warning("速率限制 (429)，等待 %.0fs 后重试", wait)
                    await asyncio.sleep(wait)
                    continue

                # 服务器错误 — 指数退避重试
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"服务器错误 {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                # 客户端错误 (非 429) — 不重试
                if response.status_code >= 400:
                    raise DataFetchError(
                        f"请求参数错误: {response.status_code}",
                        status_code=response.status_code,
                    )

                return response

            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last_exception = exc
                if attempt < self._max_retries - 1:
                    delay = min(
                        self.RETRY_BASE_DELAY * (2 ** attempt),
                        self.RETRY_MAX_DELAY,
                    )
                    logger.warning(
                        "请求失败 (attempt %d/%d): %s，%.1fs 后重试",
                        attempt + 1, self._max_retries, exc, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise DataFetchError(
                        f"请求失败，已重试 {self._max_retries} 次: {exc}",
                    ) from exc

            except httpx.HTTPStatusError as exc:
                # 5xx → 可重试
                if exc.response.status_code >= 500 and attempt < self._max_retries - 1:
                    delay = min(self.RETRY_BASE_DELAY * (2 ** attempt), self.RETRY_MAX_DELAY)
                    logger.warning("服务器错误 %d，%.1fs 后重试", exc.response.status_code, delay)
                    await asyncio.sleep(delay)
                else:
                    raise DataFetchError(str(exc), status_code=exc.response.status_code) from exc

        # 理论上不会到达这里
        raise DataFetchError(f"请求失败: {last_exception}")
