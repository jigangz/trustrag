---
title: TrustRAG v2 Completion — Latency Optimization + Real Benchmark + v1.0.0 Ship
date: 2026-04-21
status: draft
author: Jigang Zhou (Harry)
repo: github.com/jigangz/TrustRAG
scope: 4 days (compressed from 9)
predecessor: docs/superpowers/specs/2026-04-20-trustrag-v2-enhancement-design.md
---

# TrustRAG v2 Completion Design

## Executive Summary

TrustRAG v0.2 enhancement sprint is 85% complete (28/33 tasks). This spec covers the final 15% — closing P3-GATE (real benchmark numbers), P5-GATE (MCP end-to-end Claude Desktop demo), and P7-GATE (v1.0.0 release) — under a strict **$0/month cost ceiling**.

The core blocker is Railway backend latency (30-60s/query) caused by redundant embedding calls and Groq double-call architecture, compounded by Railway Hobby tier's 0.5 vCPU constraint. Rather than paying to upgrade Railway or switching embedding providers, this design **optimizes the existing architecture**:

1. **Eliminate 3 wasted fastembed invocations per query** (redundant + recomputed-already-stored embeddings)
2. **Merge Groq generation + hallucination check into one JSON-structured call** (HTTP path only; streaming preserved)
3. **Add Postgres-backed query cache** (hot queries sub-200ms)
4. **UptimeRobot keep-alive** eliminates cold starts
5. **RAGAS eval judge switched to Gemini Flash** (free 1.5M tokens/day) so Groq TPD isn't competed against benchmark

Post-optimization target: **5-10s cache miss / <200ms cache hit**, preserving Groq's streaming TTFT <500ms selling point.

Delivered in 4 workstreams across 4 days: backend opt (Day 1), benchmark (Day 2-3), MCP demo + v0.5.0-mcp (Day 3), v1.0.0 release (Day 4).

---

## 1. Context & Current State

### Completion Status (end of 2026-04-21)

```
Progress: 28 / 33 tasks (85%)
✅ P1: 6/6 (Streaming)
✅ P2: 4/4 (Hybrid)
🔄 P3: 3/4  (P3-GATE deferred — benchmark, numbers projected not measured)
✅ P4: 6/6 (LangChain)
🔄 P5: 2/4  (P5-3 + P5-GATE pending — MCP Claude Desktop demo)
✅ P6: 3/3 (n8n)
🔄 P7: 4/6  (P7-5 + P7-GATE pending — v1.0.0)
```

### Infrastructure State

- **Backend**: Railway Hobby free tier at `trustrag-production.up.railway.app` (1GB RAM, 0.5 vCPU, Postgres with pgvector via Railway template)
- **Frontend**: Vercel (009 account) at `trustrag.vercel.app`
- **PyPI published**: `trustrag-langchain 0.1.0`, `trustrag-mcp 0.1.1`, `trustrag-eval 0.1.0`
- **Data loaded**: 2 PDFs, 174 chunks via pg_dump from local
- **Current `GROQ_MODEL`**: `llama-3.1-8b-instant` (downgraded from 70B due to Railway slowness)

### Root Cause of Railway Slowness (30-60s/query)

Confirmed by reading `backend/services/*`:

| Cost source | Per query | Notes |
|------------|-----------|-------|
| Fastembed query embedding | 1 call | Necessary |
| **Fastembed redundant in `_verify_trust`** | 1 call | 🐛 `streaming.py:122` recomputes same query embedding |
| **Fastembed `_compute_source_agreement`** | 1 batch × 5 texts | 🐛 `trust_verifier.py:96` re-embeds chunks already in DB |
| Groq generation | 1 call (2-5s) | Necessary |
| Groq hallucination check | 1 call (2-5s) | Secondary LLM fact-check |
| Railway cold start | 10-20s | Free tier idle > 15 min sleeps |
| Postgres hybrid search | ~50ms | Not a bottleneck |

**Total**: ~30-60s cold / 15-25s warm. Over 50% is redundant fastembed + Groq double-call.

---

## 2. Goals & Non-Goals

### Goals (A — Strict completion)

- All 5 outstanding GATEs pass with **real measured evidence**, not projected:
  - P3-GATE: Benchmark with Groq-generated answers + Gemini-judged RAGAS metrics + hit@5 computed, written to `eval/results/`
  - P5-GATE: Claude Desktop invokes `trustrag_query` with real Railway backend, screenshot captured
  - P7-GATE: `v1.0.0` tag + GitHub Release with per-feature notes + measured benchmarks in README
- **Production URL stays online** (via UptimeRobot keep-alive, no paid uptime)
- **README shows measured numbers**, not "projected"

### Non-Goals (explicit)

- **No paid infrastructure**: Railway stays free, Groq stays free, Gemini stays free. Zero /month cost commitment.
- **No embedding provider swap**: Fastembed remains local; preserves "no external embedding API" selling point.
- **No new features**: Only optimization + release polish. v1.1+ roadmap items deferred.
- **No trust-score semantic change**: 4-factor structure preserved (retrieval 40% + source_count 20% + agreement 20% + hallucination 20%); only hallucination algorithm changes in merged HTTP path.
- **No breaking API changes**: `QueryResponse.hallucination_check.flags` contract preserved for frontend + MCP clients.

---

## 3. Locked Decisions (from brainstorm)

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| Q1 | Completion standard | A — Strict (all GATEs, production live, real numbers) | Not rushed; want portfolio-grade evidence |
| Q2 | Monthly cost ceiling | $0 | Job search budget-conscious; "optimized for free tier" is a story |
| Q3 | Embedding architecture | A — Keep fastembed + optimize code | Preserves "no external API" selling point; 3 wasted calls can be eliminated |
| Q4 | Hallucination check | A — Merged prompt (JSON structured output) in HTTP path only | Halves Groq calls; streaming path retains 2-call for critical UX |
| Q5 | RAGAS judge | B — Gemini 2.0 Flash free tier | 1.5M tokens/day sidesteps Groq TPD competition; multi-provider story |
| Extra | Groq model | `llama-3.3-70b-versatile` (env override 8B for emergency) | Benchmark number quality; daily demo volume (<20q) well under 100K TPD |

---

## 4. Architecture Changes (WS1: Backend Latency Optimization)

Five fixes in order of risk (safe → more complex). Feature-flagged for rollback safety.

### Fix 1: Eliminate redundant query embedding

**Current bug** (`backend/services/streaming.py:122`):
```python
async def _verify_trust(self, answer: str, sources: list[dict]) -> dict:
    from services.embedding import embed_text
    query_embedding = await embed_text(self.text)  # 🐛 already embedded in _retrieve!
    trust = await compute_trust_score(answer, sources, query_embedding)
```

**Fix**: Persist the embedding computed in `_retrieve` on the `QueryTask` instance and reuse in `_verify_trust`.

```python
async def _retrieve(self) -> list[dict]:
    async with async_session() as session:
        self._query_embedding = await embed_text(self.text)  # Store on self
        chunks = await hybrid_search(session, self._query_embedding, self.text, top_k=self.top_k)
    return chunks

async def _verify_trust(self, answer: str, sources: list[dict]) -> dict:
    trust = await compute_trust_score(answer, sources, self._query_embedding)  # Reuse
    # ...
```

**Savings**: 1 fastembed call (~1-2s on 0.5 vCPU).

**Risk**: None (pure local refactor, no behavior change).

### Fix 2: Reuse DB-stored embeddings in source agreement

**Current waste** (`backend/services/trust_verifier.py:96`):
```python
async def _compute_source_agreement(sources: list[dict]) -> float:
    texts = [s["content"] for s in sources[:5]]
    embeddings = await embed_batch(texts)  # 🐛 chunks.embedding column already stores these
```

**Fix**: Have `hybrid_search` (and `search_similar`) return the `embedding` column in each chunk dict, so `_compute_source_agreement` uses pre-computed vectors from upload time.

```python
# vector_store.py
async def search_similar(session, query_embedding, top_k=5) -> list[dict]:
    result = await session.execute(text("""
        SELECT c.id AS chunk_id, c.document_id, d.filename,
               c.content, c.page_number, c.embedding,   -- NEW
               1 - (c.embedding <=> :embedding) AS similarity
        FROM chunks c JOIN documents d ON c.document_id = d.id
        ORDER BY c.embedding <=> :embedding
        LIMIT :top_k
    """), {...})
    return [
        {"chunk_id": ..., "embedding": row["embedding"], ...}  # Pass through
        for row in result.mappings().all()
    ]

# trust_verifier.py
async def _compute_source_agreement(sources: list[dict]) -> float:
    if len(sources) < 2:
        return 1.0
    # Use embeddings from chunks themselves (stored in DB at upload time)
    embeddings = [s["embedding"] for s in sources[:5] if s.get("embedding") is not None]
    if len(embeddings) < 2:
        # Fallback: re-embed (shouldn't happen with updated search_similar)
        texts = [s["content"] for s in sources[:5]]
        embeddings = await embed_batch(texts)
    # ... same pairwise similarity computation
```

**Savings**: 5 fastembed calls per query (~2-5s).

**Risk**: Low. DB embedding is the same model as runtime fastembed (both `BAAI/bge-small-en-v1.5`). Fallback retains old behavior.

### Fix 3: Merged generation + self-check prompt (HTTP path only)

**Scope limit**: HTTP `/api/query/` endpoint only. Streaming path (`QueryTask.run` in `streaming.py`) keeps 2-call architecture (streaming UX handles latency of 2nd call while user reads tokens).

**New function** in `backend/services/rag_engine.py`:

```python
MERGED_SYSTEM_PROMPT = """You are a precise document assistant for construction safety.
Answer the question using ONLY the provided source documents, then perform a self-check
to identify any claims in your answer that are NOT directly supported by the sources.

Rules:
- Cite sources as [Source: document_name, p.XX]
- If sources lack sufficient information, say so explicitly
- In self_check.unsupported_claims, list any sentence that cannot be fully verified from sources

Return valid JSON matching this schema:
{
  "answer": "the answer text with inline citations",
  "self_check": {
    "unsupported_claims": [
      {"sentence": "exact sentence text", "reason": "why it's not supported"}
    ]
  }
}
"""

async def generate_answer_merged(question: str, context_chunks: list[dict]) -> dict:
    """Single-call generation + self-check via JSON structured output.
    
    Falls back to sequential generate_answer + _check_hallucination if JSON parse fails.
    """
    context = _build_context(context_chunks)
    
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": MERGED_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
        max_tokens=1500,  # Slightly higher to accommodate JSON + answer
        response_format={"type": "json_object"},
    )
    
    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        answer = parsed["answer"]
        unsupported = parsed.get("self_check", {}).get("unsupported_claims", [])
        citations = _parse_citations(answer)
        return {
            "answer": answer,
            "sources_used": citations,
            "hallucination_flags": unsupported,  # Maps to old schema
            "raw_response": response.model_dump(),
            "merged": True,
        }
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Merged JSON parse failed, falling back: %s", e)
        # Fallback: use non-merged path
        result = await generate_answer(question, context_chunks)
        flags = await _check_hallucination(result["answer"], context_chunks)
        return {
            "answer": result["answer"],
            "sources_used": result["sources_used"],
            "hallucination_flags": flags,
            "raw_response": result["raw_response"],
            "merged": False,
        }
```

**Router change** (`backend/routers/query.py`):

```python
if settings.merge_prompt_enabled:
    rag_result = await generate_answer_merged(request.question, sources)
    # Use rag_result["hallucination_flags"] directly; skip compute_trust_score's _check_hallucination
    trust_score = await compute_trust_score(
        rag_result["answer"], sources, query_embedding,
        precomputed_hallucination_flags=rag_result["hallucination_flags"],
    )
else:
    rag_result = await generate_answer(request.question, sources)
    trust_score = await compute_trust_score(rag_result["answer"], sources, query_embedding)
```

**Trust verifier signature extended** (`compute_trust_score` accepts optional precomputed flags; skips internal `_check_hallucination` call if provided).

**Savings**: 1 Groq call per HTTP query (~2-5s).

**Known tradeoff (disclosed in README)**: LLM self-check has ~5-10% bias (Llama 70B may miss its own hallucinations). Mitigation: RAGAS faithfulness metric uses Gemini as independent judge, so benchmark numbers are bias-free.

**Risk**: Medium. Groq JSON mode occasionally returns malformed JSON. Fallback to sequential path ensures no silent crashes.

### Fix 4: Postgres-backed query cache

**New table** (added to `init_db()` for Railway idempotent migration):

```sql
CREATE TABLE IF NOT EXISTS query_cache (
    question_hash TEXT PRIMARY KEY,
    response_json JSONB NOT NULL,
    hit_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_hit_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_query_cache_created ON query_cache (created_at);
```

**New service** `backend/services/cache.py`:

```python
import hashlib
import json
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

CACHE_TTL_HOURS = 24

def _hash_query(question: str, top_k: int) -> str:
    normalized = question.lower().strip()
    normalized = " ".join(normalized.split())  # Collapse whitespace
    return hashlib.sha256(f"{normalized}|{top_k}".encode()).hexdigest()

async def get(session: AsyncSession, question: str, top_k: int) -> dict | None:
    if not settings.query_cache_enabled:
        return None
    h = _hash_query(question, top_k)
    cutoff = datetime.utcnow() - timedelta(hours=CACHE_TTL_HOURS)
    result = await session.execute(
        text("SELECT response_json FROM query_cache WHERE question_hash = :h AND created_at > :cutoff"),
        {"h": h, "cutoff": cutoff},
    )
    row = result.first()
    if row:
        # Atomic increment hit counter
        await session.execute(
            text("UPDATE query_cache SET hit_count = hit_count + 1, last_hit_at = NOW() WHERE question_hash = :h"),
            {"h": h},
        )
        await session.commit()
        return row[0]
    return None

async def set(session: AsyncSession, question: str, top_k: int, response: dict):
    if not settings.query_cache_enabled:
        return
    h = _hash_query(question, top_k)
    await session.execute(
        text("""INSERT INTO query_cache (question_hash, response_json) 
                VALUES (:h, CAST(:r AS jsonb))
                ON CONFLICT (question_hash) DO UPDATE SET 
                    response_json = EXCLUDED.response_json,
                    created_at = NOW()"""),
        {"h": h, "r": json.dumps(response)},
    )
    await session.commit()

async def clear_all(session: AsyncSession):
    await session.execute(text("DELETE FROM query_cache"))
    await session.commit()
```

**Integration points**:
- `query.py` ask_question endpoint: `get()` check at start, `set()` before return
- Document upload endpoint: `clear_all()` after successful upload (to prevent stale answers)
- Benchmark runner: pass `X-Bypass-Cache: 1` header OR `?nocache=1` query param to ensure fresh runs

**New admin endpoint** `POST /admin/clear-cache` (no auth needed — portfolio, not multi-tenant):

```python
@router.post("/clear-cache")
async def clear_cache(session: AsyncSession = Depends(get_session)):
    await cache.clear_all(session)
    return {"status": "cleared"}
```

**Savings**: Cache hit → <200ms. Cache miss → same as Fix 1+2+3 pipeline.

**Audit integration**: Audit log entries get a new `from_cache BOOL` column (idempotent migration) so we can report hit rate.

**Risk**: Low. Behind `QUERY_CACHE_ENABLED` flag, default off in dev, on in prod after smoke test.

### Fix 5: UptimeRobot keep-alive + Railway model upgrade

**Setup (manual, one-time)**:
1. Register at uptimerobot.com (free, 50 monitors)
2. Add HTTP(S) monitor:
   - URL: `https://trustrag-production.up.railway.app/health`
   - Interval: 5 minutes
   - Alert contact: Harry's email
3. Railway env var update:
   ```bash
   railway variables set GROQ_MODEL=llama-3.3-70b-versatile
   ```
4. Redeploy / restart Railway service to pick up env change

**Effect**:
- 5-minute pings prevent Railway idle sleep (free tier sleeps after 15 min idle)
- 70B responses: +100-200ms TTFT but significantly higher answer quality
- UptimeRobot free tier monthly usage: ~8,640 requests/month = trivial Railway CPU impact

**Risk**: Negligible. If UptimeRobot monitoring triggers excessive CPU on Railway free tier (shouldn't, `/health` is 10ms), we lower ping frequency to 10 min.

### Summary: expected latency after WS1

| Stage | Before | After WS1 |
|-------|--------|-----------|
| Cold start | 10-20s | 0 (keep-alive) |
| Query embedding | 2-5s | 2-5s (fastembed unchanged) |
| Hybrid search | 50ms | 50ms |
| Generation + hallucination | 4-10s | 2-5s (merged, HTTP) / 4-10s (2-call, streaming) |
| Source agreement | 2-5s | 0 (DB reuse) |
| **Total cache miss (HTTP)** | **30-60s** | **5-10s** |
| **Total cache miss (streaming)** | **30-60s** | **8-15s, TTFT <500ms** |
| **Total cache hit** | N/A | **<200ms** |

---

## 5. Benchmark Execution (WS2: P3-GATE)

### Dataset Selection

`eval/synthetic_queries.json` already contains 30 hand-authored queries (10 semantic + 10 keyword + 10 hybrid). Scope for this completion is **15 queries (5 per category, the first 5 of each)**, yielding:

- Semantic (Q001-Q005): "What is the minimum height threshold for fall protection..."
- Keyword (Q011-Q015): OSHA code references
- Hybrid (Q021-Q025): Mixed semantic + keyword

Sufficient statistical signal for portfolio claims (5 per category; differences > 0.2 likely significant; stay under Groq TPD limit split across 2 days).

If time permits (stretch), run remaining 15 queries to reach full 30q corpus.

### Gemini Integration for RAGAS Judge

**Why Gemini for judge, Groq for pipeline**: Groq TPD (100K on 70B free tier) is consumed by pipeline generation. RAGAS's 4 metrics each invoke the judge LLM per query → 15 queries × 2 modes × 4 metrics × ~3K tokens = 360K tokens. Gemini 2.0 Flash free tier (1.5M tokens/day, 1500 req/day) absorbs this without competing for Groq TPD.

**Code change** in `packages/trustrag-eval/src/trustrag_eval/ragas_pipeline.py`:

```python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

def _get_gemini_judge():
    import os
    return LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0.0,
        )
    )

def _get_gemini_embeddings():
    import os
    return LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )
    )

def run_benchmark(endpoint: str, dataset_path: str, limit: int, output_path: str):
    # ... existing data gathering ...
    result = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=_get_gemini_judge(),
        embeddings=_get_gemini_embeddings(),
    )
    # ... merge with trust_metrics, write JSON ...
```

**Deps update** `packages/trustrag-eval/pyproject.toml`:
```toml
dependencies = [
    "ragas>=0.2",
    "langchain-core>=0.3",
    "langchain-google-genai>=2.0",  # NEW
    "httpx>=0.27",
    "datasets>=2.0",
]
```

**Setup steps (Day 2 morning)**:
1. Harry visits Google AI Studio, creates API key (free, ~2 min)
2. `export GOOGLE_API_KEY=...` (local), also set in Railway env (not strictly needed unless Gemini called from backend)
3. Add `GOOGLE_API_KEY` to `.env.example`
4. `pip install langchain-google-genai` in local trustrag-eval venv
5. Smoke test: 1 query through `run_benchmark(limit=1)` to verify Gemini wrapper works

### Two-Day Execution Plan

#### Day 2 (Semantic baseline)

```bash
# Clear cache before run (prevents hit from prior manual tests)
curl -X POST https://trustrag-production.up.railway.app/admin/clear-cache

# Flip semantic-only mode on Railway
railway variables set HYBRID_ENABLED=false
# Wait for redeploy

# Run semantic benchmark (15 queries, ~30 min)
cd packages/trustrag-eval
python -m trustrag_eval.ragas_pipeline \
  --endpoint https://trustrag-production.up.railway.app \
  --dataset ../../eval/synthetic_queries.json \
  --limit 15 \
  --mode semantic \
  --output ../../eval/results/2026-04-22-semantic-15q.json
```

**Groq budget**: 15 queries × 5K tokens (merged prompt) = **75K of 100K TPD** ✅.

**Gemini budget**: 15 queries × 4 metrics × ~3K = **180K of 1.5M TPD** ✅ + embedding calls ~30K = **well under limit**.

#### Day 3 morning (Hybrid)

Wait for Groq TPD UTC reset (16:00 PST or later — i.e., after a UTC day transition).

```bash
curl -X POST https://trustrag-production.up.railway.app/admin/clear-cache
railway variables set HYBRID_ENABLED=true

python -m trustrag_eval.ragas_pipeline \
  --endpoint https://trustrag-production.up.railway.app \
  --dataset ../../eval/synthetic_queries.json \
  --limit 15 \
  --mode hybrid \
  --output ../../eval/results/2026-04-23-hybrid-15q.json
```

**Groq budget**: 75K (same math) ✅.

### Results Schema

Unified JSON per run:

```json
{
  "metadata": {
    "date": "2026-04-22",
    "mode": "semantic",
    "model": "llama-3.3-70b-versatile",
    "judge_model": "gemini-2.0-flash-exp",
    "queries_count": 15,
    "endpoint": "https://trustrag-production.up.railway.app",
    "merge_prompt_enabled": true,
    "query_cache_enabled": false
  },
  "ragas": {
    "faithfulness": 0.XX,
    "answer_relevancy": 0.XX,
    "context_precision": 0.XX,
    "context_recall": 0.XX
  },
  "trust": {
    "hit_at_5": 0.XX,
    "trust_score_median": XX.X,
    "trust_score_p25": XX.X,
    "trust_score_p75": XX.X,
    "hallucination_flagged_rate": 0.XX
  },
  "per_query": [
    {"id": "Q001", "question": "...", "answer": "...", "trust_score": XX, "hit_at_5": true, ...}
  ]
}
```

### README Benchmark Table

```markdown
## 📊 Benchmarks

Measured on 15-query synthetic corpus (5 semantic + 5 keyword + 5 hybrid), Railway production deployment 2026-04-23. RAGAS metrics judged by Gemini 2.0 Flash (independent provider to avoid self-bias).

| Config | Hit@5 | Faithfulness | Ans Relevancy | Ctx Precision | Ctx Recall |
|--------|-------|--------------|---------------|---------------|------------|
| Semantic-only | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| Hybrid (RRF, k=60) | **0.XX** | **0.XX** | **0.XX** | **0.XX** | **0.XX** |
| Δ | +XX pp | +0.XX | +0.XX | +0.XX | +0.XX |

See `eval/results/2026-04-23-hybrid-15q.json` for raw data and per-query breakdown.
```

### Old Results Handling

Existing files in `eval/results/` (dated 2026-04-20, 2026-04-21) are retained but moved to `eval/results/archive/pre-optimization/` with a README.md explaining they were pre-optimization / partial runs from Ralph Loop iterations. New canonical results use 2026-04-22/23 dates.

### Release

```bash
git tag v0.3.0-hybrid -m "Hybrid retrieval: +XXpp hit@5 over semantic baseline (15q synthetic, Gemini-judged RAGAS)"
gh release create v0.3.0-hybrid --notes-file docs/releases/v0.3.0-hybrid.md
```

Release notes include: benchmark table, methodology description, link to raw JSONs, acknowledgment of LLM self-check bias disclaimer for merged path.

---

## 6. MCP End-to-End Demo (WS3: P5-GATE)

### Prerequisite

WS1 complete: Railway query latency in 5-10s range (HTTP path, merged prompt). Claude Desktop tool call timeout is 30s, so 5-10s leaves ~20s margin.

### Configuration

`~/AppData/Roaming/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "trustrag": {
      "command": "uvx",
      "args": ["trustrag-mcp"],
      "env": {
        "TRUSTRAG_BACKEND_URL": "https://trustrag-production.up.railway.app"
      }
    }
  }
}
```

(Should already be in place from Day 0. Verify after Fix 3 deploy.)

Restart Claude Desktop. Confirm `trustrag` shows as a connected MCP server with 3 tools: `trustrag_query`, `trustrag_upload_document`, `trustrag_get_audit_log`.

### Three Demo Queries

1. **trustrag_query**: Ask "What fall protection does OSHA require at 6 feet?" — Claude should invoke the tool, receive answer with trust score + citations. Screenshot the complete exchange including tool_use expansion. Save as `docs/mcp-query.png`.

2. **trustrag_upload_document**: Provide Claude a test PDF path (e.g., local sample OSHA document), ask it to add to TrustRAG. Screenshot tool invocation + success response. Save as `docs/mcp-upload.png`.

3. **trustrag_get_audit_log**: "Show me the last 5 queries that scored below 70 trust". Claude invokes audit tool with `max_trust_score=70`. Screenshot. Save as `docs/mcp-audit.png`.

**Primary README image**: `docs/mcp-demo.png` = the full trustrag_query exchange (most demo-worthy).

### Tag & Release

```bash
git tag v0.5.0-mcp -m "MCP server: 3 tools verified in Claude Desktop (trustrag_query, _upload_document, _get_audit_log)"
gh release create v0.5.0-mcp --notes-file docs/releases/v0.5.0-mcp.md
```

Release notes include: list of 3 tools with schemas, screenshots, Claude Desktop config snippet, installation `uvx trustrag-mcp` one-liner.

### Fallback Plans

If Railway latency still causes Claude Desktop timeouts despite WS1:

- **Plan A**: Point MCP at local docker (`TRUSTRAG_BACKEND_URL=http://localhost:8000`). Demo with full pipeline on local. README annotates: "MCP Claude Desktop demo captured with local backend for UX; production Railway backend works via HTTP for programmatic callers."
- **Plan B**: Use MCP Inspector (`npx @modelcontextprotocol/inspector uvx trustrag-mcp`) for screenshot evidence, noting Claude Desktop connects but Railway demo latency exceeds chat timeout threshold.

Plan A is preferred over Plan B for stronger narrative.

---

## 7. v1.0.0 Release (WS4: P7-GATE)

### README Rewrite

Sections (in order):

```markdown
# TrustRAG

[Badges row: PyPI × 3, Live Demo, GitHub Release, License, Python 3.11+]

> Trust-verified RAG platform with streaming, hybrid retrieval, and measurable quality.

## 🎯 Why TrustRAG

[3 bullets: streaming + trust + ecosystem]

## 🚀 Live Demo
- Frontend: https://trustrag.vercel.app
- Backend API: https://trustrag-production.up.railway.app
- Health: https://trustrag-production.up.railway.app/health

## 📦 Installation

[Three PyPI `pip install` commands + MCP Claude Desktop config snippet]

## 🏗️ Architecture

[ASCII diagram from design spec 2026-04-20 §2, updated with cache + keep-alive]

## 📊 Benchmarks

[Table from §5 above, measured 2026-04-23]

## 🔌 Integrations

### Claude Desktop (MCP)
![MCP Query Demo](docs/mcp-demo.png)

### LangChain Agent (Trust Budget)
[trustrag-langchain code sample]

### n8n Workflows
[3 workflow templates link + screenshots]

## 🧪 Development

[docker-compose + pytest commands]

## 📝 License
MIT
```

### `docs/releases/v1.0.0.md`

```markdown
# v1.0.0 — Production-Grade Trust-Verified RAG

Release date: 2026-04-24

Two weeks of feature development culminated in a 13-night sprint turning TrustRAG from a single-app demo into a production-deployed, ecosystem-integrated platform with measured quality.

## Highlights
- ✅ WebSocket streaming with cancellation (TTFT <500ms)
- ✅ Hybrid retrieval (pgvector + tsvector + RRF k=60) → +XXpp hit@5
- ✅ RAGAS evaluation pipeline (Gemini-judged for bias-free metrics)
- ✅ 3 PyPI packages: trustrag-langchain, trustrag-mcp, trustrag-eval
- ✅ LangGraph multi-hop agent with trust budget
- ✅ MCP server with 3 tools (tested in Claude Desktop)
- ✅ n8n workflow templates (3)
- ✅ Live deployment: Vercel + Railway

## Benchmarks (measured on 15-query synthetic corpus)

[Table]

## Packages

- `trustrag-langchain 0.1.0` — LangChain integration
- `trustrag-mcp 0.1.1` — MCP server (3 tools)
- `trustrag-eval 0.1.0` — RAGAS evaluation pipeline

## Breaking Changes
None. v0.1 API contract preserved. `hallucination_check.flags` field unchanged.

## Live URLs
- Frontend: https://trustrag.vercel.app
- Backend: https://trustrag-production.up.railway.app

## Known Tradeoffs
- HTTP `/api/query/` uses merged prompt for latency (2-call → 1-call). LLM self-check has known ~5-10% bias; RAGAS faithfulness metric (Gemini-judged) is bias-free.
- WebSocket streaming path retains 2-call architecture for stricter fact-checking under streaming UX.
- Railway Hobby free tier (1GB RAM / 0.5 vCPU): UptimeRobot keep-alive prevents cold starts; typical query latency 5-10s HTTP / 8-15s streaming.

## Roadmap
- v1.1: DOCX + HTML ingestion
- v1.2: Basic session auth + rate limiter
- v1.3: Rerank layer (cohere or local)

## Credits
Harry Zhou (Jigang Zhou). github.com/jigangz
```

### Tag

```bash
git tag v1.0.0 -m "v1.0.0: Production-grade trust-verified RAG — streaming, hybrid, 3 PyPI packages, MCP-integrated, benchmarked, deployed"
gh release create v1.0.0 --notes-file docs/releases/v1.0.0.md
```

### PyPI Version Bumps

None required — Fix 3 and Fix 4 only touch backend, not the 3 published packages. SIGN-105 (PyPI immutability) preserved.

If `trustrag-eval` gets `ragas_pipeline.py` updates for Gemini judge, bump to `0.1.1` before Day 2 benchmark run. (Alternative: run benchmark with local-source package, don't publish updated eval to PyPI until post-v1.0.0.)

### Optional Polish (Day 4 evening, if time)

- Record streaming demo as GIF: `docs/streaming-demo.gif` (ffmpeg from screen record)
- Blog draft: `docs/blog-draft-v1.md` (not published, reference material)

---

## 8. Timeline — 4 Days

| Day | Date | Workstream | Tasks | Output |
|-----|------|-----------|-------|--------|
| 1 | 2026-04-21 eve | WS1 all | Fix 1+2 (1 commit), Fix 4 cache (1 commit), Fix 3 merged (1 commit), Fix 5 infra (Railway env + UptimeRobot) | Railway backend at 5-10s; 70B; keep-alive |
| 2 | 2026-04-22 | WS2 morning + semantic | Gemini setup + wrapper; semantic 15q benchmark | `eval/results/2026-04-22-semantic-15q.json` |
| 3 | 2026-04-23 | WS2 hybrid + WS3 MCP | Hybrid 15q; v0.3.0-hybrid tag + release; 3 MCP screenshots; v0.5.0-mcp tag + release | P3-GATE ✅ P5-GATE ✅ |
| 4 | 2026-04-24 | WS4 | README rewrite; v1.0.0 release notes; tag + release | P7-GATE ✅ **SHIPPED** |

### Day 1 Breakdown (most intense)

| Time | Task | Method |
|------|------|--------|
| 2h | Fix 1 + 2 (embedding cleanup) | Ralph Loop (mechanical, low-risk) |
| 3h | Fix 4 (query cache + migration + endpoint) | Ralph Loop or manual |
| 4h | Fix 3 (merged prompt + fallback parse) | Harry manual (higher complexity; self-bias disclosure) |
| 30min | Fix 5 (Railway env + UptimeRobot) | Harry manual |
| 30min | Deploy + smoke test | Harry manual |

**Overrun mitigation**: If Fix 3 doesn't complete Day 1, defer to Day 2 morning. Day 2 benchmark can run on non-merged (2-call) pipeline — numbers still valid, just slightly slower runtime. Merged-prompt can land post-v1.0.0 as optimization.

### Dependencies

```
WS1 Fix 1+2+4+5 ──┐
                  ├─> benchmark fidelity (WS2)
WS1 Fix 3         ─┘
                  └─> MCP demo latency (WS3)
WS2 P3-GATE      ──┐
                  ├─> v1.0.0 release (WS4)
WS3 P5-GATE      ──┘
```

---

## 9. Testing Strategy

### Unit Tests (pytest per package)

**Backend**:
- `test_streaming.py::test_query_embedding_reused` — assert `_verify_trust` doesn't re-embed
- `test_trust_verifier.py::test_source_agreement_uses_db_embeddings` — fixture sources with embedding field, assert embed_batch NOT called
- `test_rag_engine.py::test_merged_prompt_schema` — mock Groq, assert JSON parse + schema match
- `test_rag_engine.py::test_merged_fallback_on_bad_json` — Groq returns non-JSON, assert fallback to separate calls
- `test_cache.py::test_hash_normalization` — "What is X?" == "what is x?" (whitespace/case-insensitive hash)
- `test_cache.py::test_ttl_expiry` — entry older than 24h not returned
- `test_cache.py::test_hit_counter_increments` — same query fetched twice, counter = 2

**trustrag-eval**:
- `test_ragas_pipeline.py::test_gemini_wrapper_loads` — assert `_get_gemini_judge()` returns LangchainLLMWrapper without error (skip if GOOGLE_API_KEY unset)

### Integration Tests

`backend/tests/test_integration_query.py`:
- Full `/api/query/` endpoint with docker-compose Postgres + mocked Groq + mocked hallucination → assert response schema preserved

### E2E

- `curl -X POST https://trustrag-production.up.railway.app/api/query/ -d '{"question": "test"}'` returns within 10s with expected fields
- `python -m trustrag_eval.ragas_pipeline --limit 1` produces valid JSON

### Manual

- Claude Desktop invocation of 3 MCP tools (screenshots)
- Frontend streaming UI still works (no regression)

### Regression

After Fix 3 merge: run 5 canonical queries through both HTTP merged path and streaming 2-call path, compare answers for consistency. Expect similar content; merged path may have slightly different self-check counts.

---

## 10. Rollout & Feature Flags

### New Environment Variables

Added to `backend/config.py` and `.env.example`:

```python
# Latency optimization flags
merge_prompt_enabled: bool = False    # Fix 3: merged generation + self-check
query_cache_enabled: bool = False     # Fix 4: postgres-backed cache

# Keep GROQ_MODEL controllable for fallback
groq_model: str = "llama-3.3-70b-versatile"  # Can override to llama-3.1-8b-instant under quota pressure

# Gemini (for benchmark eval only, not production backend)
# GOOGLE_API_KEY set only in local/CI env running benchmark runner
```

### Rollout Order

Each commit independently verifiable:

1. **Commit 1** (WS1 Fix 1+2, Day 1 morning): No flag — pure refactor, always on. Verify: `pytest backend/tests/`
2. **Commit 2** (WS1 Fix 4, Day 1 afternoon): Behind `QUERY_CACHE_ENABLED=false` default. Verify: cache tests pass, admin endpoint works.
3. **Commit 3** (WS1 Fix 3, Day 1 evening): Behind `MERGE_PROMPT_ENABLED=false` default. Verify: merged prompt unit tests pass, fallback parse tests pass.
4. **Deploy to Railway** (Day 1 late): Push main, Railway auto-deploys. Smoke-test `/health` + 1 query with both flags OFF.
5. **Enable flags on Railway** (Day 1 late): `railway variables set QUERY_CACHE_ENABLED=true MERGE_PROMPT_ENABLED=true GROQ_MODEL=llama-3.3-70b-versatile`. Smoke-test 1 query again — expect 5-10s latency.
6. **UptimeRobot setup** (Day 1 late): Configure 5-min ping.

### Rollback

If Fix 3 causes issues: `railway variables set MERGE_PROMPT_ENABLED=false`. No code revert needed.
If Fix 4 causes issues: `railway variables set QUERY_CACHE_ENABLED=false`. Cache table remains; flag just bypasses it.
If 70B quota becomes an issue: `railway variables set GROQ_MODEL=llama-3.1-8b-instant`.

---

## 11. Guardrails (New SIGN rules)

Extending SIGN-101 through SIGN-106 from predecessor design:

- **SIGN-107 (Feature flag safety)**: All new optimizations (merge_prompt, query_cache) must be behind env var flags. Default off in dev/CI; enable only after Railway verification.
- **SIGN-108 (JSON parse fallback)**: Any code parsing LLM JSON output must have try/except → fallback path. Silent crashes on malformed JSON are prohibited.
- **SIGN-109 (Cache invalidation discipline)**: Doc upload endpoint MUST clear query_cache. Benchmark runner MUST bypass cache (header or flag). Cache miss is strictly preferable to stale cache hit.
- **SIGN-110 (Self-bias disclosure)**: Merged prompt's in-line self-check is a fast-path mechanism with known LLM self-bias. README + release notes must name it explicitly; RAGAS faithfulness (Gemini-judged) is the independent reference metric. No claims of "zero hallucination" — instead "measured faithfulness 0.XX."

---

## 12. Risks

| Risk | Prob | Impact | Mitigation |
|------|------|--------|-----------|
| Fix 3 merged prompt requires > 4h on Day 1 | Med | Day 1 overrun | Defer to Day 2 morning; benchmark runs on 2-call pipeline with slightly slower numbers |
| Groq JSON mode returns broken JSON intermittently | Med | Failed queries | SIGN-108 fallback to 2-call path; log warning; continue |
| Railway keep-alive exceeds free monthly CPU budget | Low | Service sleep returns | UptimeRobot 5-min `/health` ping is ~10ms CPU × 8,640 = ~90 CPU-seconds/month, trivial |
| Gemini Flash API rate limit (1500 req/day) hit | Low | Benchmark fails | Per-day budget ~200 Gemini req; buffer >7× |
| Hit@5 Δ < 10pp on 15-query subset (stat noise) | Med | Under design spec criteria | Run additional 15 queries (days 5-6 optional) for 30q total; or disclose as "+Xpp on 15q subset" honestly |
| Claude Desktop 30s timeout triggers despite WS1 optimization | Low | P5-GATE blocked | Plan A fallback to local backend; Plan B MCP Inspector evidence |
| `init_db` query_cache migration ordering vs tsvector | Low | Deploy failure | `CREATE TABLE IF NOT EXISTS ...` idempotent; ordering irrelevant |
| Gemini `text-embedding-004` dimension mismatch with RAGAS | Low | Benchmark crashes | Embedding mismatch won't crash; RAGAS handles; test locally first |
| Self-bias causes inflated trust scores in merged path | Med | Misleading demo numbers | RAGAS faithfulness from independent Gemini judge is the disclosed reference |

---

## 13. Success Criteria (v1.0.0 ship gate)

### Quantitative

- [ ] Railway `/api/query/` latency: p50 ≤ 10s (cache miss), p95 ≤ 15s
- [ ] Cache hit latency: p50 ≤ 200ms, p95 ≤ 500ms
- [ ] WebSocket TTFT: < 500ms (preserved from v0.2 design)
- [ ] Benchmark Hit@5 Δ: ≥ +5pp (v1.0.0 stretch: +10pp; 15q subset may not hit stretch)
- [ ] RAGAS faithfulness (hybrid): ≥ 0.80
- [ ] 3 PyPI packages pip-installable (no changes needed)
- [ ] Railway + Vercel production URLs respond
- [ ] Claude Desktop invokes all 3 MCP tools successfully

### Qualitative

- [ ] README has real measured numbers (no "projected" or "TBD")
- [ ] README has at least 1 MCP Claude Desktop screenshot
- [ ] README has architecture diagram
- [ ] `docs/releases/v1.0.0.md` comprehensive release notes
- [ ] GitHub Releases page: `v0.3.0-hybrid`, `v0.5.0-mcp`, `v1.0.0` all present
- [ ] Self-bias disclosure visible in release notes

### Resume Bullets Unlocked

- "Optimized production RAG 6× (30-60s → 5-10s) by eliminating redundant embedding calls, merging Groq prompts into JSON-structured self-check, and adding Postgres-backed query cache — all within Railway free tier 1GB/0.5vCPU."
- "Benchmarked hybrid retrieval (pgvector + tsvector + RRF) vs semantic-only on 15-query synthetic corpus using RAGAS pipeline with Gemini Flash judge (independent bias-free evaluation), achieving +XXpp Hit@5 improvement."
- "Published 3 open-source PyPI packages (trustrag-langchain, trustrag-mcp, trustrag-eval) integrated into Claude Desktop via MCP server with 3 tools — demonstrated end-to-end in production."

---

## 14. Out of Scope (Explicit)

- Auth / multi-tenancy (v1.2 roadmap)
- Doc formats beyond PDF (v1.1 roadmap)
- Horizontal scaling (post v2)
- Paid tier upgrades (Railway / Groq / Gemini) — deferred indefinitely
- Fine-tuning custom models — out of product scope
- Full 30-query corpus benchmark (stretch; may run Day 5-6 if time permits, not v1.0.0 blocker)
- Mobile UI — deferred
- Rerank layer — v1.3 roadmap

---

## Appendix A: New Environment Variables

```bash
# Backend (production Railway)
MERGE_PROMPT_ENABLED=true                # Fix 3 flag
QUERY_CACHE_ENABLED=true                 # Fix 4 flag
GROQ_MODEL=llama-3.3-70b-versatile       # Was llama-3.1-8b-instant

# Benchmark runner (local only; NOT in Railway)
GOOGLE_API_KEY=AIza...                   # Gemini 2.0 Flash free tier
```

## Appendix B: New Dependencies

```toml
# packages/trustrag-eval/pyproject.toml
dependencies = [
    "ragas>=0.2",
    "langchain-core>=0.3",
    "langchain-google-genai>=2.0",   # NEW — Gemini judge wrapper
    "httpx>=0.27",
    "datasets>=2.0",
]
```

No new backend dependencies (Fix 1-5 use existing stack).

## Appendix C: API Contract Preservation

`QueryResponse` pydantic model (in `backend/models.py`) unchanged. Specifically:

```python
class HallucinationCheck(BaseModel):
    passed: bool
    flags: list[dict]

class QueryResponse(BaseModel):
    answer: str
    confidence: ConfidenceResponse
    sources: list[SourceResponse]
    hallucination_check: HallucinationCheck  # Field preserved
    consistency_check: dict | None
    audit_id: str
```

Internal: merged prompt produces `self_check.unsupported_claims`. Router maps to `hallucination_check.flags` for backwards compatibility with frontend (`frontend/src/components/TrustBadge.jsx`) and MCP response formatter (`packages/trustrag-mcp/src/trustrag_mcp/server.py`).

## Appendix D: File Change Inventory

New files:
- `backend/services/cache.py`
- `docs/releases/v0.3.0-hybrid.md`
- `docs/releases/v0.5.0-mcp.md`
- `docs/releases/v1.0.0.md`
- `docs/mcp-query.png`
- `docs/mcp-upload.png`
- `docs/mcp-audit.png`
- `docs/mcp-demo.png` (primary)
- `eval/results/2026-04-22-semantic-15q.json`
- `eval/results/2026-04-23-hybrid-15q.json`
- `eval/results/archive/pre-optimization/README.md`

Modified files:
- `backend/services/streaming.py` — reuse query_embedding
- `backend/services/trust_verifier.py` — accept precomputed hallucination_flags, use DB embeddings for source agreement
- `backend/services/vector_store.py` — search_similar + hybrid_search include embedding column
- `backend/services/rag_engine.py` — new generate_answer_merged function
- `backend/routers/query.py` — cache integration, merged branch
- `backend/routers/documents.py` — cache invalidation on upload
- `backend/config.py` — 2 new flags
- `backend/database.py` — init_db adds query_cache table
- `backend/models.py` — (possibly add from_cache to audit schema; optional)
- `.env.example` — GOOGLE_API_KEY + new flags
- `packages/trustrag-eval/src/trustrag_eval/ragas_pipeline.py` — Gemini judge wrapper
- `packages/trustrag-eval/pyproject.toml` — langchain-google-genai dep
- `README.md` — full rewrite per §7

Files moved/archived:
- `eval/results/2026-04-20-*.json` → `eval/results/archive/pre-optimization/`
- `eval/results/2026-04-21-*.json` → `eval/results/archive/pre-optimization/`

---

*End of v2 Completion Design — 2026-04-21.*
