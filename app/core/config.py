"""应用配置（pydantic-settings，从 .env 读取）。

对外统一入口：
    from app.core.config import settings            # 既有模块（RAG / Runtime）
    from app.core.config import get_settings        # Sentiment & Risk Agent（同源兼容入口）

字段名与 .env 环境变量（大写、下划线）自动对应，例如：
    RAG_VECTOR_STORE_PATH  →  rag_vector_store_path
环境变量优先级高于 .env 文件。
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置。默认值供未配置 .env 时兜底。"""

    # ── RAG ──────────────────────────────────────────────────
    # 向量库持久化根目录（各 company 独立子目录：<path>/<company>/）
    rag_vector_store_path: str = "data/vector_store"

    # Research Memory checkpoint 后端（PR39+PR41）：memory | sqlite | postgres
    #   memory   → LangGraph MemorySaver（进程内存活）
    #   sqlite   → SqliteSaver（本地磁盘，跨进程重启可恢复）
    #   postgres → AsyncPostgresSaver（PR41 生产级，多 worker/多任务并行）
    rag_checkpoint_store: str = "memory"
    # SqliteSaver 数据库文件（仅 rag_checkpoint_store=sqlite 时用）
    rag_checkpoint_db_path: str = "data/checkpoints.db"

    # ── PostgreSQL（PR41）──────────────────────────────────────
    # 业务库 DSN（research_tasks / research_reports 表）。
    database_url: str = ""
    # LangGraph checkpoint DSN（langgraph schema 下的 checkpoint 表）。
    # 可与 business 同库，仅 search_path 隔离。例：
    #   postgresql://eduagent_user:123456@localhost:5433/finance_agent
    checkpoint_db_url: str = ""

    # ── Runtime Reliability（PR42a Worker Recovery）────────────
    # worker 心跳周期（秒）：worker 定期上报存活（heartbeat_at）。
    runtime_heartbeat_interval: float = 10.0
    # lease 过期阈值（秒）：heartbeat_at 超时判 stale，可被接管。
    # 必须大于最坏单节点同步阻塞时长（设计文档 §4c），默认 30s。
    runtime_lease_ttl: float = 30.0
    # reaper 扫描周期（秒）：孤儿接管 + watchdog 巡检。
    runtime_reaper_interval: float = 15.0
    # watchdog 单代次上限（秒）：从 claimed_at 起算，超时判 failed。
    # 用 claimed_at 而非 heartbeat_at —— heartbeat 一直在跳，回答不了"跑了多久"。
    runtime_max_run_seconds: float = 600.0
    # 单任务最大认领代次：超过判 failed，防无限 crash 循环。
    runtime_max_attempts: int = 3
    # worker 标识（默认 hostname:pid；多进程部署可显式覆盖）。
    runtime_worker_id: str = ""

    # ── Runtime Worker Pool（PR43 Redis 任务队列 + 多 Worker）─────
    # 并发 worker 数量：控制同时执行的研究任务上限（调度/并发节流）。
    runtime_worker_count: int = 4
    # Redis 连接 URL（任务调度队列；PR43 只用于"哪个任务需要执行"的调度，
    # 任务状态仍以 PostgreSQL research_tasks 为准）。
    runtime_redis_url: str = "redis://localhost:6379"
    # Redis 队列 key（RPUSH 入队 / BLPOP 出队的 list）。
    runtime_redis_queue_key: str = "finance:research:queue"
    # 优雅关闭超时（秒）：shutdown 等待在途任务完成，超时强制取消（PR43.5 ③）。
    runtime_shutdown_timeout: float = 30.0

    # retrieve() 默认返回条数（精排后）
    rag_default_top_k: int = 5

    # Hybrid 召回候选数（供 CrossEncoder 精排；对应 retriever 的粗排 top_k）
    # Top-K sweep 实测：20 候选准确率 100% 不掉，CrossEncoder 延迟近减半（vs 50）
    rag_retrieve_top_k: int = 20

    # 精排后返回条数（retriever 默认 k）
    rag_rerank_top_k: int = 5

    # 精排模型：dummy（默认，直通无模型依赖）/ metadata（MetadataReranker，
    # CrossEncoder + 元数据信号融合）/ 模型路径（如 app/models/reranker/bge-reranker-v2-m3）
    rag_reranker_model: str = "dummy"

    # ── Metadata Rerank 融合权重（PR #33）────────────────────────
    # final = α*ce_norm + β*section_signal + γ*keyword_signal
    # 按公司配置：{"小米": "0.90,0.08,0.02"}。未配置公司 → 纯 CrossEncoder（零回归）。
    # 语义信号对干净文本语料（如 CATL）已最优，metadata 增强只对分析师查询 + OCR 受损
    # 语料（如小米）有增益（MRR 0.26→0.42）。
    rag_metadata_company_weights: dict[str, str] = {}

    # Embedding 模型：bge-m3（默认，本地 BGE-M3）/ dummy（开发占位，无 torch 依赖）
    embedding_model: str = "bge-m3"

    # Embedding 推理设备：auto（默认，GPU 优先无则 CPU）/ cuda / cpu（无 CUDA torch 时强制）
    rag_embedding_device: str = "auto"

    # Hybrid 检索：Dense(BGE-M3) + Sparse(BM25) + RRF 融合（默认开启）
    rag_hybrid: bool = True

    # PDF OCR fallback：文本层损坏（PyMuPDF/pdfplumber 均乱码）时用 Tesseract（默认开启）
    rag_ocr: bool = True

    # ── Vector Backend（PR44.4 Production Vector Backend Switch）───
    # 生产向量后端：faiss（默认）/ milvus。切换需先跑 migrate + verify（PR44.3.2/3）。
    # FAISS 保留用途：local development / offline evaluation / emergency recovery。
    # 回滚：改回 faiss 重启即可（FAISS 数据从不删除）。
    rag_vector_backend: Literal["faiss", "milvus"] = "faiss"

    # ── Milvus（PR44.3.1 Milvus Adapter）─────────────────────────
    # 仅 get_store(backend="milvus") 时使用；pymilvus 为可选依赖（懒加载）。
    # Milvus 服务地址（生产独立部署；开发 Milvus Lite "./milvus.db" 或 Docker）。
    milvus_uri: str = "http://localhost:19530"
    # Collection 名（AD-2 finance_knowledge；所有公司共享，company_id 字段隔离）。
    milvus_collection_name: str = "finance_knowledge"
    # 向量维度（BGE-M3=1024；collection 创建时固定，非运行时推断）。
    milvus_dim: int = 1024
    # Milvus Database（AD-1 冻结 = finance_agent；禁止传 "" 或 default —— 共享环境只新建不碰他库）。
    milvus_db_name: str = "finance_agent"

    # ── Multi-source Fusion（PR #35）─────────────────────────────
    # 参与多源知识融合的文档语义类型（SourceType 枚举值）。
    # 默认仅年报；加入研报/政策等文档入库后在此扩展。
    rag_source_types: list[str] = ["annual_report"]

    # ── Research Intent（PR #36）────────────────────────────────
    # 意图解析模式：rule（默认，规则关键词，确定性）/ llm（DeepSeek，留作增强）/ off（直通兜底）
    rag_research_parser: str = "rule"

    # 已知公司列表（意图目标抽取用；含标准名，别名见 intent.py _COMPANY_ALIASES）
    rag_known_companies: list[str] = ["小米", "宁德时代", "小鹏汽车"]

    # ── Query Rewrite（PR #31）─────────────────────────────────
    # 查询改写模式：rule（默认，规则同义词扩展，零成本）/ llm（LLM 生成变体）/ off（直通）
    rag_query_rewriter: str = "rule"

    # LLM 改写：DeepSeek（OpenAI 兼容 API），密钥从 .env 的 DEEPSEEK_API_KEY 读取
    llm_rewrite_base_url: str = "https://api.deepseek.com"
    llm_rewrite_model: str = "deepseek-v4-flash"
    llm_rewrite_temperature: float = 0.5
    llm_rewrite_max_queries: int = 3  # LLM 生成的额外变体数（不含原始 query）

    # ── DeepSeek LLM（Sentiment & Risk Agent，PR Feature sentiment-risk-agent）──
    # 密钥从 .env 的 DEEPSEEK_API_KEY 读取；未配置时 Sentiment/Risk Agent 走 Mock LLM 降级。
    deepseek_api_key: str = ""
    # OpenAI 兼容接口地址（llm_factory 统一使用；注意带 /v1 路径）。
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # ── Auth（JWT 双 Token，PR 新增）─────────────────────────────
    # 签名密钥，从 .env 的 JWT_SECRET_KEY 读取（已在 .env 中）。
    jwt_secret_key: str = ""
    # 签名算法（HS256 为对称默认；改动需同时更新前端无感知）。
    jwt_algorithm: str = "HS256"
    # access token 有效期（分钟，短命，前端 401 自动 refresh）。
    access_token_expire_minutes: int = 15
    # refresh token 有效期（天，长命，用于续签 access）。
    refresh_token_expire_days: int = 7
    # token 版本声明名（存于 JWT payload；bump 用户 token_version 即服务端吊销）。

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # .env 含项目其他模块的键（POSTGRES_* / GITEE_TOKEN 等），非 RAG 键一律忽略
        extra="ignore",
    )


settings = Settings()


def get_settings() -> Settings:
    """返回全局配置单例（与模块级 ``settings`` 同源）。

    Sentiment & Risk Agent 兼容入口：
        from app.core.config import get_settings
        s = get_settings()
        api_key = s.deepseek_api_key
    """
    return settings
