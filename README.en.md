# FinaceAgent · Multi-Agent Financial Intelligence Analysis Platform

> **[English](README.en.md)｜[中文](README.md)**
>
> A multi-agent investment-research pipeline driven by **LLM + LangGraph + RAG + real financial data + a Chinese sentiment model + a React frontend**. Feed it a target company; it produces a six-chapter structured report.

---

## 1. The Problem

Traditional investment research = **collect → read → organize → quantify → judge risk → write the report**. It has three structural pain points:

1. **Scattered sources**: annual reports, filings, news, and industry data live all over the place.
2. **Stale timeliness**: markets and sentiment require real-time tracking that humans cannot sustain.
3. **Repetitive, error-prone work**: metric calculation and formatting are highly repetitive, and when "computing" and "explaining" are mixed it becomes hard to audit.

FinaceAgent is not meant to be "a smarter chatbot". It is meant to **replicate a real research team with several single-purpose agents** — let programs do what is automatable (retrieval, computation, extraction, attribution), let models do what needs judgment (interpretation, weighing, grading), and keep everything **auditable, degradable, reusable**.

---

## 2. Design Philosophy

### 2.1 Single-responsibility agents — mirroring a real team
The system maps an offline team to **Manager + 5 specialized agents**. **One agent does one thing**; cross-domain capability duplication is forbidden (an architectural invariant):

| Agent | Responsibility | Data source |
|---|---|---|
| Manager | intent, planning, orchestration | LLM |
| Research | fundamentals / industry | RAG knowledge base |
| Financial | metric computation | structured API |
| Sentiment | news & public sentiment | news retrieval + NLP |
| Risk | 3-dimension assessment | signals from upstream agents |
| Report | report generation | all upstream outputs |

### 2.2 Classify data by nature — structured via Tool, unstructured via RAG
**Reject "dump everything into RAG"**. Layer the data by nature:

```
Structured (indicators/quotes) → Tool + hard compute (Pandas)
Unstructured knowledge (annual reports/policy) → RAG (chunk → embed → retrieve → attribute)
Time-sensitive (news/sentiment) → real-time retrieval + sentiment classification
```

### 2.3 Strict separation of compute and explanation (no numeric hallucination)
**"The model explains; it does not compute."** ROE, DuPont, YoY are hard-computed by `quant_engine` with Pandas; the LLM only interprets the pre-computed metrics. This removes numeric errors from the model side entirely.

### 2.4 Anti-hallucination — evidence attribution (Research Agent)
Report generation uses **evidence index attribution**: the LLM only returns the **indices** of the evidence it cites (`evidence_refs`); the real `source / page / quote` is bound afterward from a `ref_map`. Out-of-range indices are silently dropped; claims without evidence are kept but flagged as "evidence empty". **Citations cannot be fabricated by the model.**

### 2.5 Balance determinism vs intelligence — decouple layer by layer
- **Deterministic layer**: ReportAssembler **deterministically** builds six chapters (zero-parse / zero-dependency); LLM only polishes a few spots.
- **Intelligent layer**: nodes / intent / planning go to the LLM.
- **Runaway protection**: every loop has a **cap** (health-retry ≤2, report-quality ≤3, RAG adaptive ≤3). The cap is the sole forced exit — no infinite loops, no runaway cost.

### 2.6 Fault tolerance first — every agent has a degradation path
- Research: real-RAG unavailable (no ingestion / model load failure) → graceful placeholder, **never breaks the main chain**.
- Financial: akshare fails → local fixture → empty.
- Sentiment: news API fails → placeholder; model unavailable → neutral.
- Risk: missing data → rule-based grading.
- The health-check loop detects unhealthy nodes and retries them.

### 2.7 Observability & reuse
- **Observable**: SSE per-node progress, system readiness (DB / Milvus / models / data sources, item by item).
- **Reusable**: every agent result is persisted; report generation **automatically reuses the latest history** to avoid duplicate work.

---

## 3. Architecture Thinking

### 3.1 Overall topology — a "plan → parallel → gather → refine" main chain

```
                    ┌──────────────┐
                    │  Manager      │  intent & planning
                    └──────┬───────┘
                           │ intent_router  ← conditional (full_research / clarify)
               ┌───────────┼───────────┐  Send API fan-out (true parallelism)
               ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Research │ │Financial │ │Sentiment │
        │ (RAG)    │ │ (akshare)│ │ (news+NLP)│
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             └────────────┼────────────┘   ← gather
                          ▼
                 ┌──────────────┐  ┌─ Loop 1: health-check retry (≤2, re-send only failed nodes)
                 │ health_check │──┘
                 └──────┬───────┘   → risk (all healthy / retries exhausted; degrade inside node)
                        ▼
                 ┌──────────────┐
                 │   Risk       │  3-dimension assessment
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐  ┌─ Loop 2: report-quality iteration (≤3, append revision)
                 │   Report     │──┘
                 └──────┬───────┘
                        ▼
                 evaluate_report → report / clarify (ask for more info)
```

### 3.2 Key architecture decisions

**AD-1: Orchestration = LangGraph StateGraph + Send API, not a serial pipeline.**
The three agents (Research / Financial / Sentiment) fan out **in parallel** via `Send`, then gather — conditional edges take on intent routing and the two loops. This is closer to real team collaboration than "call 5 functions in order", and state stays a `TypedDict` (traceable / serializable / recoverable).

**AD-2: A VectorStore abstraction — FAISS / Milvus switchable.**
`app/rag/vectorstore/get_store(company, backend)` is the single entry; `RAG_VECTOR_BACKEND` switches the backend. FAISS is for local / offline eval / emergency fallback; Milvus for production (`finance_agent.finance_knowledge`). Multi-company isolation via a scalar `company_id` filter (AD-7), not partitions.

**AD-3: Reliability = "two loops + full-chain degradation", not a single try/except.**
- **Health-check retry loop**: guards the quality of parallel-agent outputs; unhealthy nodes are **only the failed ones re-sent** (avoid repeating side-effecting actions), with a cap.
- **Report-quality iteration loop**: scores the final report (section completeness / leftover placeholders / length) and appends a revision if it fails.
- Every loop has a **single forced exit** (the cap) → guaranteed finite convergence.

**AD-4: Anti-hallucination via evidence index attribution, not "trust the LLM".**
Each claim keeps only `evidence_refs` (indices); real sources are bound afterward from the ref_map — citations cannot be forged by the model and are auditable.

**AD-5: Real data sources + Chinese sentiment — no demo placeholders.**
- Financial: akshare **EM annual reports** (exchange-prefixed symbols, absolute yuan, latest fiscal year).
- News: akshare EastMoney `stock_news_em`.
- Sentiment: **Chinese 3-class model** (distilled multilingual), instead of English-only FinBERT that returns neutral for Chinese.

**AD-6: Single entry + per-agent independent trigger.**
One `uvicorn app.main:app` process mounts every route; each agent also has its own endpoint (Financial / Research / Sentiment / Risk / Knowledge / History / Report / SSE / status) — so you can debug a single module in isolation or run the whole chain.

**AD-7: GPU-aware.**
BGE-M3 and the reranker auto-detect CUDA (`RAG_EMBEDDING_DEVICE=auto`); BGE runs **fp16** to halve VRAM so the reranker can share an 8 GB GPU (otherwise it gets pushed to CPU and becomes ~6× slower). Falls back to CPU when no GPU.

---

## 4. Key Highlights

1. **Real multi-agent orchestration**: `Send` API parallel fan-out + conditional branching + **two loops** (health retry / quality iteration), each capped to avoid runaway.
2. **Anti-hallucination RAG**: evidence-index attribution + Dense(BGE-M3)+Sparse(BM25)→RRF→CrossEncoder re-ranking.
3. **Compute/explain separation**: metrics hard-computed in Pandas, model only interprets — no numeric hallucination.
4. **Real data + Chinese sentiment**: akshare real reports/news + Chinese 3-class sentiment model — no placeholders.
5. **Single entry multi-agent**: one command runs everything; each agent is independently triggerable.
6. **Offline ingestion**: upload from UI → live SSE progress → GPU-first embedding → vector store.
7. **History persistence + auto reuse**: report generation reuses the latest history instead of re-running.
8. **Observable React workbench**: dark/light theme, per-agent views, SSE full-chain visualization, system readiness.

---

## 5. End-to-End Data Flow

```
User input (company / ticker / question)
 → Manager plans → intent_router
 → parallel【Research RAG ∥ Financial hard-compute ∥ Sentiment news+sentiment】
 → health_check(retry loop) → Risk 3-dim → Report six chapters
 → evaluate_report(quality loop) → report (Markdown/HTML) + live SSE progress
```

## 6. Tech Stack

| Module | Technology |
|---|---|
| Language | Python 3.11+ / TypeScript |
| Backend | FastAPI · Uvicorn |
| Agent orchestration | LangGraph (StateGraph + Send API + conditional edges + loops) |
| Database | PostgreSQL (business tables + LangGraph checkpoint) · Redis |
| Vector store | FAISS (offline/fallback) · Milvus v2.4 (production, `finance_agent.finance_knowledge`) |
| Retrieval | BGE-M3 (1024-d, GPU fp16) · BM25 · RRF · CrossEncoder (metadata re-rank) |
| Financial data | akshare (EM reports + news) |
| Sentiment | Chinese 3-class multilingual model |
| Auth | PyJWT + passlib[bcrypt] (access/refresh) |
| Frontend | React 19 · Vite · TS · recharts · react-markdown |
| Deployment | Docker Compose (postgres16/redis7/etcd/minio/milvus/attu) |

## 7. Structure

```
FinaceAgent/
├── app/
│   ├── main.py            # single-entry FastAPI (all routes)
│   ├── api/               # auth/financial/research/sentiment/risk/report/
│   │                      #   knowledge/history/system_status/analyze_stream
│   ├── agents/            # agent implementations (financial/sentiment/risk)
│   ├── workflow/          # LangGraph State + Graph (main chain + two loops)
│   ├── rag/               # RAG: load/split/embed/vectorstore/retrieve/rerank/
│   │                      #   research agent/evaluation
│   ├── report/            # ReportAssembler (6 chapters) + exporter (MD/HTML)
│   ├── database/          # async SQLAlchemy session
│   ├── models/            # ORM/Pydantic (users/agent_runs/sentiment_risk)
│   ├── services/          # orchestration (auth/agent_history/data_fetcher)
│   ├── tools/             # @tool wrappers (news/sentiment/risk)
│   ├── core/              # config/security(JWT)/schemas/llm_factory
│   └── quant_engine/      # hard financial computation engine
├── frontend/              # React Vite TS workbench
├── scripts/               # init_db/seed_users/migrate_faiss_to_milvus/verify_milvus/demo
├── evaluation/            # RAG evaluation & results
├── docker/                # postgres init / redis conf
├── docs/                  # design docs
├── docker-compose.yml     # infra orchestration (6 services)
└── tests/                 # unit/regression tests
```

## 8. Quick Start

### Prerequisites
- Python 3.11+ (conda: torch / sentence-transformers / pymilvus / akshare)
- Node 20+ · Docker · `.env` (`DEEPSEEK_API_KEY` / `DATABASE_URL` / `MILVUS_URI` / `JWT_SECRET_KEY`)

### Infra + init + seed
```bash
docker compose up -d
/venv/bin/python scripts/init_db.py
/venv/bin/python scripts/seed_users.py
```

### Run
```bash
/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000   # backend (single entry)
cd frontend && npm install && npm run dev                              # frontend :5173
```
Sign in: `admin/admin123` (admin), `analyst/analyst123`, `demo/demo123`.

## 9. API Overview (`/api/v1`)

| Endpoint | Purpose |
|---|---|
| `POST /auth/login` `POST /auth/refresh` `GET /auth/me` | auth |
| `POST /financial` | Financial standalone |
| `POST /research/analyze` | Research standalone (SSE node progress; fast/deep) |
| `POST /sentiment` `POST /risk` `POST /sentiment-risk/full` | sentiment / risk / combined |
| `GET /knowledge/companies` `GET /knowledge/search` `POST /knowledge/upload` | knowledge base |
| `POST /history` `GET /history` | history save / list (reused by report) |
| `POST /report/generate` | six-chapter report (auto-reuses history) |
| `POST /analyze` `POST /analyze/stream` | main chain / SSE full chain |
| `GET /health/status` | system readiness |

## 10. Testing & Quality
- `tests/test_*.py`: unit / regression (routing / ingestion / milvus / retriever ...).
- `--run-real`-marked tests need real models (BGE-M3 / Milvus) and are skipped by default.
- Evaluation baselines: Xiaomi R@5=80% / MRR=0.423, CATL R@5=100% (`evaluation/`).

## 11. Known Limitations

### 11.1 Data-source applicability (important)
> The current financial / news adapters (akshare: EM annual reports + EastMoney news) are **built for listed A-share companies (Shanghai / Shenzhen)**.

- **Unlisted companies (no ticker)**:
  - **Financial data limited**: the `Financial` agent relies on akshare's listed-company financials. For an unlisted company → **no data** → falls back to a local fixture / empty, so the "financial analysis" chapter is empty or degraded. The **fundamentals / industry analysis (Research RAG) still works** — just ingest its annual report / prospectus / policy docs.
  - In other words: **querying "quotes / financials" presupposes the stock is listed with public report data.**
- **Hong Kong stocks (akshare limitation)**:
  - The current adapters treat the code as an **A-share** (`SZ` / `SH` prefix). A **HK code (e.g. 00700 Tencent) is treated as an A-share** → the EM annual-report endpoint returns nothing / news is empty.
  - To support HK stocks you'd need akshare's **HK-specific interfaces** (`stock_hk_*`) plus HK-prefix / exchange routing.
- **A-share code convention**: `_prefixed_symbol` adds `SH` for codes starting with `6/5/9`, otherwise `SZ` (e.g. `300750 → SZ300750`).

> 📄 Data-source / interface list & integration guide: see [`docs/data_source_notes.md`](docs/data_source_notes.md).

### 11.2 Other known limitations
- **Research is slow**: adaptive multi-step retrieval + LLM synthesis — fast mode ~1-2 min, deep 3-6 min. Retrieval itself is GPU-accelerated (re-rank CPU→GPU, ~6×).
- **Sentiment model**: neutral sentences are judged weakly in Chinese; positive / negative are accurate (ProsusAI/finbert is English-only and deprecated).
- **Network**: DeepSeek / HuggingFace / akshare need internet or a proxy.
- **Dual entry**: `app.main:app` is the current single entry; `app.api.app:create_app` (Research RAG task system) is retained, not yet merged.

## 12. Roadmap
- Frontend UX polish (validation / error messages)
- Further research fast-mode speed-ups (fewer candidates / fewer steps)
- Productionization: auth hardening (refresh rotation/revocation), multi-tenancy, PDF export
- More data sources & Chinese sentiment fine-tuning

## License

[MIT](LICENSE) © 2026 FinaceAgent contributors.

---

> This project is for educational / research purposes; output is for reference only and is not investment advice.
