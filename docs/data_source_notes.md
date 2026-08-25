# FinaceAgent · 数据源与接口说明（Data Source Notes）

> 本文件说明系统当前**接入了哪些数据源 / 接口**、**代码如何约定**、以及**能查什么、不能查什么**（适用性边界）。读者用它判断某个公司 / 市场是否可分析。

---

## 1. 已接入的数据源

| 类别 | 来源 | 接口函数 | 覆盖 |
|---|---|---|---|
| 财务 / 行情 | akshare（东方财富 EM）| `ak.stock_profit_sheet_by_report_em` 、`ak.stock_balance_sheet_by_report_em` | **A 股/沪深**上市公司 |
| 新闻舆情 | akshare（东方财富）| `ak.stock_news_em` | **A 股**关注（部分其他市场可能为空）|
| 情感 | 多语言三分类模型 | `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | 中文（正/负较准，中性句偏弱）|
| 基本面知识 | RAG（BGE-M3）| 本地嵌入 → FAISS/Milvus `finance_knowledge` | 任意公司（**只要入库**）|
| LLM | DeepSeek | `deepseek-v4-flash`（OpenAI 兼容）| 意图 / 规划 / 解读 / 润色 |

> ⚠️ akshare 的 EM 财务/新闻接口**只面向 A 股代码**；港股、美股需另接 akshare 的 `stock_hk_*` / 对应接口。全部数据源均需**联网或走代理**。

---

## 2. 代码约定：股票代码 → 交易所前缀

`app/services/data_fetcher.py` 的 `_prefixed_symbol(ticker)`：

```python
def _prefixed_symbol(ticker: str) -> str:
    return ("SH" + ticker) if ticker.startswith(("6", "5", "9")) else ("SZ" + ticker)
```

- `300750 → SZ300750`（深市）
- `600519 → SH600519`（沪市）
- 其它市场（港股 `00700`、美股）会被当作 **A 股**处理 → 查询失败/为空。

**取数逻辑**（`_fetch_akshare_sync`）：
- 利润表：`TOTAL_OPERATE_INCOME`（营收）、`PARENT_NETPROFIT`（归母净利）
- 资产负债表：`TOTAL_ASSETS`、`TOTAL_PARENT_EQUITY`（归母权益）
- 只取**年报**（`REPORT_DATE` 含 `12-31`）最新的四指标，绝对元口径。
- **任一字段缺失/解析失败 → 返回空**，上层回退本地 fixture（绝不产部分脏数据）。

---

## 3. 适用边界（能查什么 / 不能查什么）

### 3.1 未上市公司（无股票代码）

- **财务受限**：`Financial` Agent 依赖 akshare 的**上市财报**。未上市公司 → `data_fetcher` 无 akshare 数据 → 回退本地 `app/data/{ticker}_*.json`（仅示例公司有）→ 否则**空**，「财务分析」章节为空或降级。
- **基本面/行业不受影响**：`Research` Agent 走 RAG，**只要把该公司的年报/招股书/政策文档入库**即可检索。未上市公司能分析基本面，但**拿不到结构化财务指标**。

> **结论**：能查"行情 / 财务指标"的前提 = **该股已上市且有公开财报**；未上市公司只能做基于入库文档的基本面/行业/舆情/风险分析。

### 3.2 港股（akshare 限制）

- 当前 `_prefixed_symbol` 只处理 A 股；港股代码会被加 `SZ`/`SH` 前缀 → EM 年报接口**查不到**、`stock_news_em` 新闻可能为空。
- **要支持港股**需：
  1. 接 akshare `stock_hk_*` 系列（港股财务/行情）；
  2. 代码加 **HK 前缀**与交易所路由（如 `00700 → HK00700`）；
  3. 新闻源同理适配。

### 3.3 A 股（完整可用）

- 沪深 A 股 **已上市** 公司：财务四指标 + 新闻 + 情感 + 基本面 RAG（入库后）**全链路可用**。

---

## 4. 接入新数据源（扩展指南）

| 目标 | 位置 | 改哪 |
|---|---|---|
| 新增财务/行情源 | `app/services/data_fetcher.py` | `_fetch_from_akshare` / 新增源方法，映射到 `FinancialMetric` |
| 新增新闻源 | `app/tools/news_tools.py` | `fetch_recent_news`，映射到 `NewsItem` |
| 换情感模型 | `app/tools/sentiment_tools.py` | `_FINBERT_MODEL_NAME` + `_LABEL_ALIAS` |
| 换向量后端 | `app/rag/vectorstore/` | `get_store`（`RAG_VECTOR_BACKEND`）|

**通用要求**：所有外部能力封装为职责单一的 `@tool`；结构化数据走 Tool，非结构化走 RAG；任一源异常→**降级**（fixture/占位/中性），不中断主链；字段映射必须与 `app/core/schemas.py` 对齐。

---

## 5. 相关配置（`.env`）

| 键 | 说明 |
|---|---|
| `RAG_VECTOR_BACKEND` | `milvus`（生产）/ `faiss`（本地/降级）|
| `RAG_EMBEDDING_DEVICE` | `auto`（GPU 优先）/ `cpu` |
| `MILVUS_URI` | Milvus 地址（如 `http://localhost:19531`）|
| `DEEPSEEK_API_KEY` | LLM 密钥 |
| `DATABASE_URL` / `CHECKPOINT_DB_URL` | PostgreSQL（业务 / checkpoint）|
