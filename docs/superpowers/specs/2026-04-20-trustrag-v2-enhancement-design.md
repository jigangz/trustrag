---
title: TrustRAG v0.2 — Streaming, Hybrid, Agents, and MCP
date: 2026-04-20
status: draft
author: Jigang Zhou (Harry)
repo: github.com/jigangz/TrustRAG
scope: 13 nights (high-intensity sprint)
---

# TrustRAG v0.2 Enhancement Design

## Executive Summary

Upgrade TrustRAG from a single-app demo into a **production-grade, ecosystem-integrated trust-verified RAG platform** with:

- **WebSocket streaming** (multi-stage status, cancellation)
- **Hybrid retrieval** (pgvector semantic + Postgres tsvector keyword + RRF fusion)
- **Automated eval pipeline** (RAGAS + trust-specific metrics on 30-query synthetic benchmark)
- **3 published PyPI packages** (`trustrag-langchain`, `trustrag-mcp`, `trustrag-eval`)
- **LangGraph multi-hop agent with trust budget**
- **MCP server** (3 tools, stdio transport; works in Claude Desktop / Cursor / Claude Code)
- **n8n workflow templates** (doc ingestion, Slack trust-gated Q&A, daily low-confidence digest)
- **Live deployment** (Vercel frontend + Railway backend with pgvector)

### Primary Goal
Build a portfolio-grade project that produces concrete resume/interview ammunition:
- Measurable improvement numbers (retrieval hit@5, faithfulness, etc.)
- Live demo URL
- Published OSS packages with download metrics
- GitHub Releases with feature-level notes

### Non-Goals
- Multi-tenancy / user auth beyond basic session
- Horizontal scaling / HA
- Additional document formats (only PDF stays)
- Mobile app
- Fine-tuning / training custom models

---

## 1. Context & Motivation

### Current State (v0.1)

TrustRAG v0.1 is a working but basic RAG demo:
- FastAPI backend + React frontend, Docker Compose
- Semantic-only retrieval via pgvector + fastembed (local embeddings)
- Groq Llama 3.3 70B for answer generation and trust verification
- 4-factor trust score (retrieval similarity, source count, source agreement, hallucination check)
- 3x rephrase consistency check
- Full audit logging
- Blocking `/api/query` returns full JSON; users wait 3–8s staring at a spinner
- No integrations, no public deployment, no measurable benchmarks

### Problems Being Solved

1. **Perceived latency**: Users wait for the full pipeline (retrieve + generate + verify + consistency) before seeing anything.
2. **Retrieval quality is brittle on keyword-heavy queries**: Pure embedding-based retrieval misses exact-match needs (e.g., "OSHA 1926.501").
3. **No quantitative quality signal**: "Trust verification" is narrative; no reproducible numbers demonstrate effectiveness.
4. **Isolation from ecosystem**: TrustRAG cannot plug into LangChain agents, Claude Desktop, or automation workflows.
5. **Weak recruiter signal**: Local-only demo, no public packages, no live URL.

---

## 2. High-Level Architecture

### System Diagram

```
                  ┌─────────── 3 Client Entrypoints ──────────┐
                  │                                            │
            ┌─────▼─────┐   ┌──────────┐   ┌─────────────────┐
            │ React WS  │   │ LangChain│   │ Claude Desktop/ │
            │ Frontend  │   │  Agent   │   │ Cursor (MCP)    │
            └─────┬─────┘   └─────┬────┘   └────────┬────────┘
                  │               │                  │
             WebSocket       HTTP REST         JSON-RPC stdio
                  │               │                  │
                  └───────────────┼──────────────────┘
                                  │
                         ┌────────▼─────────┐
                         │ FastAPI Backend  │
                         └────────┬─────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
      ┌─────▼──────┐    ┌─────────▼────────┐   ┌───────▼─────────┐
      │  pgvector  │    │  tsvector (BM25) │   │  Groq Llama 3.3 │
      │ (semantic) │    │    (keyword)     │   │   (generation)  │
      └─────┬──────┘    └─────────┬────────┘   └───────┬─────────┘
            │                     │                    │
            └─────────┬───────────┘                    │
                      │                                │
               ┌──────▼──────┐                         │
               │ RRF Fusion  │◄────────────────────────┘
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │Trust Engine │  (4 factors)
               └──────┬──────┘
                      │
                ┌─────▼──────┐
                │ Audit Log  │
                └────────────┘
```

### Repo Structure (Multi-Package Mono-Repo)

```
trustrag/
├── backend/                     # Existing + streaming + hybrid
│   ├── main.py
│   ├── routers/
│   │   ├── query.py             # WebSocket + legacy HTTP
│   │   ├── documents.py
│   │   └── audit.py
│   ├── services/
│   │   ├── rag_engine.py
│   │   ├── vector_store.py      # + hybrid_search()
│   │   ├── ranking.py           # 🆕 RRF
│   │   ├── streaming.py         # 🆕 WS handler + QueryTask
│   │   └── trust_verifier.py
│   └── tests/
│
├── frontend/                    # Existing + WS client
│   ├── src/
│   │   ├── components/
│   │   │   ├── StreamingAnswer.jsx   # 🆕
│   │   │   └── TrustBadge.jsx
│   │   └── hooks/
│   │       └── useWebSocketQuery.js  # 🆕 replaces useQuery
│
├── packages/                    # 🆕 3 PyPI packages
│   ├── trustrag-langchain/
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── src/trustrag_langchain/
│   │   │   ├── retriever.py     # L1: BaseRetriever
│   │   │   ├── tool.py          # L2: BaseTool wrapper
│   │   │   └── agent.py         # L3: LangGraph + trust budget
│   │   └── tests/
│   │
│   ├── trustrag-mcp/
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── src/trustrag_mcp/
│   │   │   ├── server.py        # 3 tools, stdio transport
│   │   │   └── client.py        # httpx client to backend
│   │   └── tests/
│   │
│   └── trustrag-eval/
│       ├── pyproject.toml
│       ├── README.md
│       ├── src/trustrag_eval/
│       │   ├── ragas_pipeline.py
│       │   ├── trust_metrics.py # TrustRAG-specific metrics
│       │   └── dataset.py
│       └── tests/
│
├── integrations/
│   └── n8n/
│       └── workflows/
│           ├── doc-ingestion.json
│           ├── slack-ask-trust-gate.json
│           └── daily-low-confidence-digest.json
│
├── eval/
│   ├── synthetic_queries.json   # 30 hand-authored Q&A
│   ├── run_benchmark.py
│   └── results/                 # timestamped JSON results
│
├── docs/
│   ├── architecture.md
│   ├── benchmarks.md
│   └── superpowers/specs/
│       └── 2026-04-20-trustrag-v2-enhancement-design.md   # this file
│
├── plans/                       # Ralph Loop orchestration
│   ├── prd.json
│   ├── guardrails.md
│   └── progress.md
│
├── scripts/ralph/
│   ├── ralph.sh
│   ├── ralph-stop.sh
│   ├── ralph-status.sh
│   └── ralph-tail.sh
│
├── .github/
│   └── workflows/
│       ├── test.yml
│       ├── publish-pypi.yml     # on tag pkg-*-v* → publish
│       └── frontend-build.yml
│
├── docker-compose.yml
├── .env.example
└── README.md                    # New hero + badges + demo GIF
```

### 13-Night Phase Map

| Night | Phase | Feature | GATE? |
|-------|-------|---------|-------|
| 1 | Streaming 1 | WS backend: `/api/ws` endpoint + message schema + QueryTask state machine | |
| 2 | Streaming 2 | Frontend WS client + multi-stage UX + cancel + error handling | ✅ **GATE** |
| 3 | Hybrid 1 | Alembic migration (tsvector + GIN); `hybrid_search()` in `vector_store.py` | |
| 4 | Hybrid 2 | `ranking.py` RRF implementation; 30-query synthetic dataset authored | |
| 5 | Eval | RAGAS pipeline in `trustrag-eval`; run benchmark; numbers committed to `eval/results/` | ✅ **GATE** |
| 6 | LangChain 1 | `trustrag-langchain` package: `retriever.py` (L1) + `tool.py` (L2) + tests | |
| 7 | LangChain 2 | `agent.py` (L3): LangGraph multi-hop agent with trust budget state | |
| 8 | LangChain 3 | `pyproject.toml` + GitHub Action + **publish `trustrag-langchain==0.1.0` to PyPI** | ✅ **GATE** |
| 9 | MCP 1 | `trustrag-mcp` package: server with 3 tools (`query`, `upload_document`, `get_audit_log`) | |
| 10 | MCP 2 | `pyproject.toml` + PyPI publish + **real Claude Desktop integration test** | ✅ **GATE** |
| 11 | n8n | 3 workflow JSON templates + canvas screenshots + README import guide | |
| 12 | Polish | README hero rewrite + badges + demo GIF + blog draft + `trustrag-eval` to PyPI | |
| 13 | Deploy | Vercel frontend + Railway backend + `v1.0.0` GitHub Release with notes | ✅ **GATE** |

---

## 3. WebSocket Streaming Protocol (Phase 1)

### Connection Lifecycle

Client upgrades to WS at `/api/ws`. Server creates per-connection state (user_id, active_queries map, last_activity timestamp). First server frame: `{"type": "connected", "server_version": "..."}`.

### Message Schema

**Client → Server:**

```json
// Start a query
{
  "type": "query",
  "id": "q_01J7XK2R",
  "text": "What's fall protection requirement at 6ft?",
  "top_k": 5,
  "min_trust_score": 0
}

// Cancel an in-progress query
{"type": "cancel", "id": "q_01J7XK2R"}

// Post-answer feedback (written to audit log)
{
  "type": "feedback",
  "id": "q_01J7XK2R",
  "rating": "good",
  "comment": ""
}
```

**Server → Client:**

```json
// Lifecycle
{"type": "connected", "server_version": "0.2.0"}

// Multi-stage progress (WS vs SSE key differentiator)
{"type": "status", "id": "q_...", "stage": "retrieving"}
{"type": "status", "id": "q_...", "stage": "generating"}
{"type": "status", "id": "q_...", "stage": "verifying_trust"}

// Streaming content
{"type": "sources", "id": "q_...", "sources": [{"doc":"...","page":3,"similarity":0.87}]}
{"type": "token", "id": "q_...", "content": "The"}

// Finalization
{"type": "trust", "id": "q_...", "score": 87, "breakdown": {...}}
{"type": "consistency", "id": "q_...", "score": 0.92, "rephrases_matched": 3}
{"type": "done", "id": "q_...", "audit_id": "a_..."}

// Errors
{"type": "error", "id": "q_...", "code": "GROQ_RATE_LIMIT", "message": "...", "retry_after_ms": 2000}
{"type": "cancelled", "id": "q_..."}
```

### Server State Machine

```
QUEUED → RETRIEVING → GENERATING → VERIFYING → COMPLETE → (audit)
             │              │             │
             └─── any ─────>│── CANCELLED (via cancel message)
                                   │
                              partial tokens still written to audit
```

### Cancellation Implementation

```python
# backend/services/streaming.py
class QueryTask:
    def __init__(self, query_id: str):
        self.id = query_id
        self.cancelled = asyncio.Event()

    async def run(self, text: str, ws):
        await self._emit_stage(ws, "retrieving")
        sources = await self.hybrid_search(text)
        if self.cancelled.is_set(): return await self._emit_cancelled(ws)

        await self._emit_sources(ws, sources)
        await self._emit_stage(ws, "generating")
        async for token in self.generate_stream(sources, text):
            if self.cancelled.is_set(): return await self._emit_cancelled(ws)
            await ws.send_json({"type": "token", "id": self.id, "content": token})

        await self._emit_stage(ws, "verifying_trust")
        trust = await self.verify_trust(...)
        # ...
```

Cancel flag is an `asyncio.Event`; every stage boundary checks before proceeding. Cancel message → set event → next checkpoint emits `cancelled` frame and returns.

### Keepalive + Reconnect

- **Keepalive**: WS protocol-level ping/pong via uvicorn `ws_ping_interval=20`. No application-layer heartbeat.
- **Reconnect**: Client listens for `onclose`; reconnects with exponential backoff `1s → 2s → 4s → 8s (max)`. On reconnect, queries are not automatically resumed (v1 requires manual retry; v2 may resume via audit_id).

### Error Codes

| Code | Meaning | Retryable | Client Behavior |
|------|---------|-----------|-----------------|
| `GROQ_RATE_LIMIT` | Groq 429 | ✅ backoff by `retry_after_ms` | Show "rate limited, retrying..." |
| `DOC_NOT_FOUND` | Query references deleted doc | ❌ | Show error |
| `TRUST_VERIFY_FAILED` | Secondary LLM failed | ❌ | Show answer with "trust score unavailable" |
| `INVALID_QUERY` | Empty/too long | ❌ | Frontend should validate |
| `INTERNAL` | Unknown | ❌ | Show generic error |

### Frontend State

```javascript
// frontend/src/hooks/useWebSocketQuery.js
const [state, setState] = useState({
  connectionStatus: 'disconnected',  // disconnected | connecting | connected
  queryStatus: 'idle',               // idle | retrieving | generating | verifying | done | cancelled | error
  answer: '',                        // accumulated tokens
  sources: [],
  trust: null,
  consistency: null,
  error: null,
});

const { sendQuery, cancelQuery, sendFeedback } = useWebSocketQuery();
```

---

## 4. Hybrid Retrieval + RRF (Phase 2)

### Migration

```sql
ALTER TABLE chunks
ADD COLUMN content_tsv tsvector
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX idx_chunks_tsv ON chunks USING GIN (content_tsv);
```

`GENERATED ALWAYS AS ... STORED` auto-computes tsvector on insert/update. Postgres 12+.

### Retrieval Flow

```
Query
  │
  ├──► Path A: Semantic (pgvector cosine) ───┐
  │    top-20 candidates                      │
  │                                           ▼
  └──► Path B: Keyword (tsvector ts_rank) ──► RRF Fusion (k=60) ──► top-5
       top-20 candidates
```

Two paths run in parallel via `asyncio.gather`.

### SQL

**Semantic:**
```sql
SELECT id, content, document_id, page,
       1 - (embedding <=> $1::vector) AS similarity
FROM chunks
ORDER BY embedding <=> $1::vector
LIMIT 20;
```

**Keyword:**
```sql
SELECT id, content, document_id, page,
       ts_rank(content_tsv, plainto_tsquery('english', $1)) AS rank
FROM chunks
WHERE content_tsv @@ plainto_tsquery('english', $1)
ORDER BY rank DESC
LIMIT 20;
```

### RRF Implementation

```python
# backend/services/ranking.py
def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (Cormack et al. 2009).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

k=60 is the empirical optimum from the RRF paper; relatively insensitive to k.

### Hybrid Search Entry Point

```python
# backend/services/vector_store.py
async def hybrid_search(self, query: str, top_k: int = 5) -> list[Chunk]:
    query_vec = await embed_async(query)
    semantic, keyword = await asyncio.gather(
        self._semantic_search(query_vec, limit=20),
        self._keyword_search(query, limit=20),
    )
    fused = rrf_fuse([
        [c.id for c in semantic],
        [c.id for c in keyword],
    ])
    top_ids = [doc_id for doc_id, _ in fused[:top_k]]
    return await self._fetch_chunks_by_ids(top_ids)
```

### Config (new)

```python
# backend/config.py
hybrid_enabled: bool = True        # env override for rollback
semantic_candidates: int = 20
keyword_candidates: int = 20
rrf_k: int = 60
final_top_k: int = 5
```

Setting `hybrid_enabled=False` falls back to pure `_semantic_search`; zero-risk rollback.

### Edge Cases

| Case | Behavior |
|------|----------|
| Query all stopwords | Keyword path returns empty; fall back to semantic only |
| Query too short (<2 chars) | Frontend validates and rejects |
| Both paths empty | Return `[]`; Trust Engine marks `source_count=0` → low trust score |
| Same chunk in both rankings | RRF scores add, naturally boosted (intentional) |
| `hybrid_enabled=False` | Use `_semantic_search` unchanged |

---

## 5. RAGAS Evaluation Pipeline (Phase 3)

### Synthetic Dataset

30 hand-authored queries in `eval/synthetic_queries.json`:
- 10 **semantic-heavy** ("How do I prevent falls?")
- 10 **keyword-heavy** ("OSHA 1926.501")
- 10 **hybrid** ("How to anchor a harness when working at height?")

Schema:
```json
{
  "queries": [
    {
      "id": "Q001",
      "text": "What's the fall protection height threshold in construction?",
      "category": "semantic",
      "ground_truth_chunk_ids": ["c_042", "c_087"],
      "expected_answer_substring": "6 feet"
    }
  ]
}
```

### Metrics

From RAGAS:
- **Faithfulness**: Does the answer stay within retrieved context? (Hallucination detection)
- **Answer Relevancy**: Does the answer actually address the question?
- **Context Precision**: How relevant are the retrieved chunks?
- **Context Recall**: Did we retrieve the chunks needed for the ground truth?

Custom (TrustRAG-specific, in `trust_metrics.py`):
- **Trust Score Distribution** (median, p25, p75)
- **Hallucination Flagged Rate** (% with trust < 50)
- **Hit@5**: Ground-truth chunk in top-5?

### Benchmark Runner

```python
# packages/trustrag-eval/src/trustrag_eval/ragas_pipeline.py
def run_benchmark(endpoint="http://localhost:8000") -> dict:
    dataset = load_synthetic_queries()
    rows = []
    for q in dataset:
        resp = httpx.post(f"{endpoint}/api/query", json={"question": q.text})
        data = resp.json()
        rows.append({
            "question": q.text,
            "answer": data["answer"],
            "contexts": [s["content"] for s in data["sources"]],
            "ground_truth": q.expected_answer,
            "trust_score": data["trust_score"],
        })
    result = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    result.update(compute_trust_metrics(rows))
    return result
```

### Reporting

Results committed to `eval/results/YYYY-MM-DD-<label>.json`. README includes comparison table:

| Configuration | Hit@5 | Faithfulness | Answer Relevance | Context Precision |
|---------------|-------|--------------|------------------|-------------------|
| Semantic-only | 73% | 0.81 | 0.85 | 0.72 |
| Hybrid (RRF) | 89% | 0.87 | 0.88 | 0.84 |

(Numbers TBD after actual run.)

---

## 6. LangChain L3 Agent (Phase 4)

### Three Levels

**L1 — BaseRetriever** (`retriever.py`):
```python
class TrustRAGRetriever(BaseRetriever):
    endpoint: str = "http://localhost:8000"
    min_trust_score: int = 70

    def _get_relevant_documents(self, query: str) -> list[Document]:
        resp = httpx.post(f"{self.endpoint}/api/query", json={"query": query})
        data = resp.json()
        if data["trust_score"] < self.min_trust_score:
            return []  # Filter out untrusted answers
        return [Document(
            page_content=data["answer"],
            metadata={
                "trust_score": data["trust_score"],
                "sources": data["sources"],
            },
        )]
```

**L2 — BaseTool** (`tool.py`):
Wraps the retriever for use in LangChain agents:
```python
class TrustRAGTool(BaseTool):
    name: str = "trustrag_query"
    description: str = "Query the trust-verified knowledge base."
    retriever: TrustRAGRetriever

    def _run(self, query: str) -> str:
        docs = self.retriever.get_relevant_documents(query)
        # ... format as string with trust annotations
```

**L3 — LangGraph Agent with Trust Budget** (`agent.py`):

State schema:
```python
class TrustBudgetState(TypedDict):
    question: str
    subqueries: list[str]
    retrievals: list[dict]          # {query, answer, trust_score, sources}
    cumulative_trust: float
    min_threshold: int               # default 150
    max_retrievals: int              # default 3
    final_answer: str | None
```

Graph:
```
START → retrieve → decide ──┬── retrieve (loop)
                             ├── answer → END
                             └── stop_low_trust → END
```

Routing:
```python
def route_after_decide(state: TrustBudgetState) -> str:
    if len(state["retrievals"]) >= state["max_retrievals"]:
        return "answer" if state["cumulative_trust"] >= state["min_threshold"] else "stop_low_trust"
    if state["cumulative_trust"] >= state["min_threshold"]:
        return "answer"
    return "retrieve"
```

Outcomes:
| Outcome | Trigger | Returned |
|---------|---------|----------|
| `answer` | Cumulative trust ≥ 150 | Synthesized answer + citation chain across all retrievals |
| `stop_low_trust` | Max retrievals reached, trust < 150 | "Insufficient evidence (cumulative trust: 120/150). Please consult a domain expert." |
| `error` | Retriever exception | Error with cause |

---

## 7. MCP Server (Phase 5)

### 3 Tools

```python
# packages/trustrag-mcp/src/trustrag_mcp/server.py
from mcp.server import Server
from mcp.types import Tool

app = Server("trustrag-mcp")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="trustrag_query",
            description="Ask the TrustRAG knowledge base. Returns answer with trust score (0-100). Filters below min_trust_score.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "min_trust_score": {"type": "integer", "default": 0, "maximum": 100},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="trustrag_upload_document",
            description="Upload a PDF to the knowledge base. Returns document_id after processing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="trustrag_get_audit_log",
            description="Fetch recent query audit entries. Useful for reviewing low-trust queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                    "max_trust_score": {"type": "integer"},
                },
            },
        ),
    ]
```

### Transport

**stdio** (JSON-RPC over stdin/stdout). Recommended by MCP spec for local integrations. Works with `uvx` for ephemeral execution.

### Claude Desktop Config

```json
// ~/AppData/Roaming/Claude/claude_desktop_config.json (Windows)
{
  "mcpServers": {
    "trustrag": {
      "command": "uvx",
      "args": ["trustrag-mcp"],
      "env": {
        "TRUSTRAG_BACKEND_URL": "http://localhost:8000"
      }
    }
  }
}
```

### Night 10 GATE

In Claude Desktop, ask a question that triggers `trustrag_query`. Verify:
1. Claude invokes the tool (check tool_use message)
2. Response includes trust score
3. Screenshot saved to `docs/mcp-demo.png`

---

## 8. n8n Workflows (Phase 6)

### Three Templates

1. **`doc-ingestion.json`** — Google Drive folder watch → POST `/api/documents/upload` → Slack notify on success/fail
2. **`slack-ask-trust-gate.json`** — Slack `/ask` slash command → POST `/api/query` → IF `trust_score >= 70` reply to user; ELSE route to `#review-queue`
3. **`daily-low-confidence-digest.json`** — Cron (6am daily) → GET `/api/audit?max_trust=60&since=24h` → summarize → email admin

Each workflow:
- JSON file in `integrations/n8n/workflows/`
- Canvas screenshot in `docs/screenshots/`
- README with import instructions and required env vars

---

## 9. PyPI Packaging

### pyproject.toml Template

```toml
[project]
name = "trustrag-langchain"
version = "0.1.0"
description = "LangChain integration for TrustRAG — trust-verified retrieval + multi-hop agent with trust budget"
authors = [{ name = "Jigang Zhou", email = "zjg0907008@gmail.com" }]
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
dependencies = [
    "langchain-core>=0.3",
    "langgraph>=0.2",
    "httpx>=0.27",
    "pydantic>=2",
]

[project.urls]
Homepage = "https://github.com/jigangz/TrustRAG"
Repository = "https://github.com/jigangz/TrustRAG"
Issues = "https://github.com/jigangz/TrustRAG/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Tag + Publish Convention

Tag format: `pkg-<name>-v<version>` (e.g., `pkg-langchain-v0.1.0`).

```yaml
# .github/workflows/publish-pypi.yml
on:
  push:
    tags:
      - "pkg-langchain-v*"
      - "pkg-mcp-v*"
      - "pkg-eval-v*"

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Parse tag
        id: parse
        run: |
          TAG="${GITHUB_REF#refs/tags/pkg-}"
          NAME="${TAG%-v*}"
          VER="${TAG#*-v}"
          echo "pkg_dir=trustrag-$NAME" >> $GITHUB_OUTPUT
          echo "version=$VER" >> $GITHUB_OUTPUT
      - name: Build
        run: |
          cd packages/${{ steps.parse.outputs.pkg_dir }}
          pip install build
          python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: packages/${{ steps.parse.outputs.pkg_dir }}/dist/
```

### Publish Schedule

- **Night 8 GATE**: `pkg-langchain-v0.1.0` → PyPI
- **Night 10 GATE**: `pkg-mcp-v0.1.0` → PyPI
- **Night 12**: `pkg-eval-v0.1.0` → PyPI

---

## 10. Deployment (Phase 7)

### Frontend → Vercel

1. `vercel --cwd frontend`
2. Root directory: `frontend`, framework: Vite
3. Env var: `VITE_API_BASE_URL=<Railway backend URL>`
4. Custom domain optional
5. URL: `https://trustrag.vercel.app`

### Backend + Postgres → Railway (primary)

1. Create Railway project
2. Add Postgres using Railway's official pgvector template
3. Connect GitHub repo, root = `backend`
4. Env vars: `GROQ_API_KEY` (manual), `DATABASE_URL` (auto-injected by Railway)
5. Deploy; URL: `https://trustrag-backend.up.railway.app`

### Post-Deploy

- Upload 3 public-domain PDFs (OSHA construction safety documents)
- Run 5 smoke queries to verify
- Record production benchmark: `eval/results/YYYY-MM-DD-production.json`
- Tag `v1.0.0` and write GitHub Release notes

### Backup Plan (Render + Supabase)

If Railway $5 trial credit runs low:
- Backend: Render free tier (Web Service)
- Postgres: Supabase free tier (pgvector enabled in dashboard)
- Add README disclaimer: "Demo may take ~30s to cold start"

---

## 11. Testing Strategy

### Unit Tests

Each package has its own `tests/` and pytest config.

- `backend/tests/test_streaming.py` — QueryTask state machine, cancellation flow (mock WS)
- `backend/tests/test_ranking.py` — RRF correctness (known inputs/outputs)
- `backend/tests/test_hybrid_search.py` — Integration: fixture DB + query → expected ranking
- `packages/trustrag-langchain/tests/test_retriever.py` — Mock backend HTTP; verify trust filtering
- `packages/trustrag-langchain/tests/test_agent.py` — LangGraph state transitions, budget logic
- `packages/trustrag-mcp/tests/test_server.py` — Tool schemas, call dispatching
- `packages/trustrag-eval/tests/test_ragas_pipeline.py` — Sample dataset → expected metric shape

### Integration

`docker-compose.yml` starts full stack; integration tests hit live Postgres + Groq (with separate test API key or mocked).

### E2E (Phase GATEs)

- Night 2 GATE: WS end-to-end (connect → query → token stream → trust → done)
- Night 5 GATE: Benchmark runner produces valid JSON in `eval/results/`
- Night 8 GATE: `pip install trustrag-langchain` from PyPI, import, invoke
- Night 10 GATE: Claude Desktop invokes `trustrag_query`, screenshot saved
- Night 13 GATE: Live Railway URL responds to query

### Ralph verifyCommand

```bash
cd backend && pytest tests/ -v && ruff check . && \
cd ../packages/trustrag-langchain && pytest tests/ -v && \
cd ../trustrag-mcp && pytest tests/ -v && \
cd ../trustrag-eval && pytest tests/ -v
```

---

## 12. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Groq rate limit during benchmark (30 queries × 4 RAGAS eval calls each) | Med | Delay | Add retry backoff; cache responses during dev |
| PyPI name collision | Low | Rename needed | Check availability before Night 6 |
| Railway $5 credit depletes pre-job-offer | Med | Demo offline | Fallback plan (Render + Supabase) ready |
| Ralph Loop `passes:true` false positive | Med | Broken feature merged | Phase GATE always manual + audit agent |
| MCP spec changes mid-project | Low | MCP package rewrite | Pin `mcp` dep version; watch release notes |
| LangGraph API changes | Low | Agent rewrite | Pin `langgraph>=0.2,<0.3` |
| RAGAS misrepresents actual quality | Med | Misleading numbers | Always pair RAGAS numbers with synthetic hit@5 |

---

## 13. Success Criteria

### Quantitative
- Hit@5 improvement from hybrid ≥ +10pp (target 73% → 89%)
- TTFT reduced to < 500ms (from 3–8s)
- 3 PyPI packages published with working `pip install`
- Benchmark committed to repo with reproducible script
- Live Railway URL accessible

### Qualitative
- Claude Desktop can invoke TrustRAG via MCP and return trust-annotated answer
- Demo GIF in README shows streaming + trust badge
- README has badges (stars, last commit, license, Python version, build status)
- GitHub Releases page has `v0.2.0`, `v0.3.0`, ..., `v1.0.0` with per-feature notes
- Blog draft written (may or may not be published during sprint)

### Resume Bullets (post-ship)

- Built production RAG with streaming, hybrid retrieval (pgvector + tsvector + RRF), and built-in trust verification.
- Published 3 open-source PyPI packages: `trustrag-langchain` (LangGraph multi-hop agent with trust budget), `trustrag-mcp` (MCP server, 3 tools), `trustrag-eval` (RAGAS wrapper).
- Hybrid retrieval (pgvector + Postgres tsvector + RRF) improved hit@5 from X% → Y% on 30-query synthetic benchmark.
- Multi-stage WebSocket streaming with query cancellation reduced TTFT from 3–8s to < 500ms.
- LangGraph multi-hop agent with cumulative trust budget terminates early when evidence insufficient, preventing confident hallucinations in agentic workflows.

---

## 14. Guardrails (Ralph SIGN rules)

Copy from big-project-workflow universal template, plus TrustRAG-specific:

- **SIGN-101: Never Bypass Trust Verification**: Trust verification is non-negotiable in the pipeline. Every `/api/query` response must include a trust score and audit entry. Do not add a "skip trust" flag.
- **SIGN-102: Hybrid Fallback Must Work**: `hybrid_enabled=False` must preserve pre-v0.2 behavior. Test both paths.
- **SIGN-103: WebSocket Backpressure**: Use `await ws.send_json(...)` to respect backpressure; never queue unbounded.
- **SIGN-104: RAGAS Cost Awareness**: Each RAGAS eval makes 4+ LLM calls per query. For 30 queries = 120+ calls. Cache and batch.
- **SIGN-105: PyPI Version Immutability**: Never re-upload same version. Bump patch version and re-tag if build fails post-publish.
- **SIGN-106: MCP Tool Names Are Public**: Once a tool name ships to PyPI, it's a breaking change to rename. Settle naming before first publish.

---

## 15. Out of Scope (Explicit Non-Goals)

- User authentication (beyond optional session token)
- Multi-tenant isolation
- Horizontal scaling / HA
- Additional doc formats (DOCX, HTML) — PDF only
- Custom embedding models / fine-tuning
- Mobile app
- Real-time collaboration (Discord-style multi-user chat)
- Billing / payments
- Admin dashboard beyond existing audit UI
- Internationalization (English only for v1)

---

## Appendix A: Full Database Schema (post-migration)

```sql
-- chunks table (existing + tsvector column)
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    page INTEGER,
    chunk_index INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chunks_tsv ON chunks USING GIN (content_tsv);
CREATE INDEX idx_chunks_document ON chunks (document_id);

-- audit_log (existing)
CREATE TABLE audit_log (
    id UUID PRIMARY KEY,
    query_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    sources JSONB,
    trust_score INTEGER,
    trust_breakdown JSONB,
    consistency_score FLOAT,
    feedback_rating TEXT,         -- 'good' | 'bad' | NULL
    feedback_comment TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Appendix B: Environment Variables (full list)

```bash
# Backend
GROQ_API_KEY=gsk-...
DATABASE_URL=postgresql://user:pass@host:5432/trustrag
GROQ_MODEL=llama-3.3-70b-versatile
LLM_MODEL=llama-3.3-70b-versatile
HYBRID_ENABLED=true
SEMANTIC_CANDIDATES=20
KEYWORD_CANDIDATES=20
RRF_K=60
FINAL_TOP_K=5

# MCP server (when used standalone)
TRUSTRAG_BACKEND_URL=http://localhost:8000

# Frontend
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/api/ws

# CI/CD
PYPI_API_TOKEN=pypi-...   # GitHub secret
```

---

## Appendix C: Key External Dependencies (new)

| Package | Version | Purpose |
|---------|---------|---------|
| `mcp` | latest stable | Anthropic's MCP Python SDK |
| `langchain-core` | ≥0.3 | Base interfaces for retriever/tool |
| `langgraph` | ≥0.2 | Graph-based agent runtime |
| `ragas` | latest | RAG evaluation metrics |
| `hatchling` | latest | PEP 517 build backend |

---

*End of design.*
