"""RAGAS evaluation pipeline with Gemini-as-judge for TrustRAG benchmarks.

v2-completion (2026-04-22): Gemini 2.5 Flash drives RAGAS LLM judge
(faithfulness / answer_relevancy / context_precision / context_recall) and
embeddings. Keeps Groq TPD free for pipeline generation per SIGN-104 /
v2 completion design §5.

Note: gemini-2.0-flash was removed from the free tier (limit:0 as of
2026-04). gemini-2.5-flash is the current free-tier Flash model — ~1500
req/day + generous TPD, more than enough for 15q × 4 metrics.

Self-bias note: the TrustRAG backend's HTTP path uses merged in-prompt
self-check (SIGN-112). RAGAS `faithfulness` is the independent bias-free
metric — both numbers are reported but not directly comparable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

import httpx
from datasets import Dataset
from ragas import RunConfig, evaluate
# NOTE: we use the legacy underscored modules intentionally. The newer
# `ragas.metrics.collections.*` classes require a modern InstructorLLM
# (via llm_factory) and are NOT accepted by `ragas.evaluate()` as of
# ragas 0.4.3 (evaluate checks isinstance(m, Metric), but collections
# use a different base class). Legacy imports are backward-compat +
# accept LangchainLLMWrapper via the `llm=` / `embeddings=` kwargs to
# evaluate(). Deprecation warnings are expected and tolerated.
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

from trustrag_eval.dataset import Query, load_synthetic_queries
from trustrag_eval.trust_metrics import compute_trust_metrics


logger = logging.getLogger(__name__)

BATCH_SIZE = 5
BATCH_PAUSE_SECONDS = 10


# ---------------------------------------------------------------------------
# Judge + embeddings (v2-completion)
# ---------------------------------------------------------------------------
# Provider strategy:
#   - Groq Llama 3.1 8B Instant is the DEFAULT judge: 500K TPD free tier,
#     ~500 tok/s inference, perfect for RAGAS's binary (1/0) verdict prompts.
#   - Gemini 2.5 Flash Lite is a fallback: free but AFC internal retries
#     loop for 30-40s/call, making 60-job RAGAS run take 2+ hours.
#   - Embeddings use Gemini (gemini-embedding-001): Groq has no embedding
#     API, and Gemini embedding free tier is plenty for RAGAS's ~15 embed
#     calls per benchmark.


def _get_groq_judge():
    """Build RAGAS-compatible LLM wrapper around Groq Llama 3.1 8B Instant.

    Uses OpenAI-compatible endpoint + ChatOpenAI langchain wrapper.
    Model: llama-3.1-8b-instant — 500K TPD free tier (5x the 70B quota),
    500+ tok/s inference, good enough for RAGAS binary verdicts.

    Raises RuntimeError if GROQ_API_KEY not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get free at https://console.groq.com "
            "(same key used for pipeline generation; reuse Railway value)"
        )

    from langchain_openai import ChatOpenAI

    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError:
        from ragas.llms.base import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOpenAI(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.0,
            timeout=30,
            max_retries=3,
        )
    )


def _get_gemini_judge():
    """Build RAGAS-compatible LangchainLLMWrapper around Gemini 2.0 Flash.

    Raises RuntimeError if GOOGLE_API_KEY not set. Gemini free tier
    (1M tokens/day, 1500 req/day) is plenty for 15q × 4 metrics benchmarks.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Get one free at "
            "https://aistudio.google.com/app/apikey and `export GOOGLE_API_KEY=...`"
        )

    # Late imports so module can be loaded for code inspection without deps
    from langchain_google_genai import ChatGoogleGenerativeAI

    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError:
        from ragas.llms.base import LangchainLLMWrapper  # older ragas layout

    # Using gemini-2.5-flash-lite on free tier: higher RPM than 2.5-flash
    # (~30 RPM vs 10) which matters because RAGAS bursts 10-15 LLM calls per
    # query × 15 queries. gemini-2.0-flash has limit:0 on free tier (Google
    # removed it 2026-04), gemini-2.5-flash hits RPM wall quickly. Lite trades
    # a bit of judge quality for reliable throughput — acceptable since RAGAS
    # uses multi-vote / statement-level aggregation.
    return LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=api_key,
            temperature=0.0,
        )
    )


def _get_gemini_embeddings():
    """Build RAGAS-compatible embeddings wrapper (text-embedding-004)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Get one free at "
            "https://aistudio.google.com/app/apikey and `export GOOGLE_API_KEY=...`"
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        from ragas.embeddings.base import LangchainEmbeddingsWrapper  # older layout

    return LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
    )


# ---------------------------------------------------------------------------
# Backend querying (cache-bypassing for benchmark fidelity per SIGN-111)
# ---------------------------------------------------------------------------

async def _query_endpoint(
    client: httpx.AsyncClient,
    endpoint: str,
    query: Query,
    max_retries: int = 5,
) -> dict:
    """Query the TrustRAG backend with exponential backoff on 429.

    Adds nocache=True per SIGN-111 so benchmark measures real pipeline,
    not cache hits. Uses trailing slash to match Railway FastAPI mount.
    """
    url = f"{endpoint.rstrip('/')}/api/query/"
    payload = {"question": query.text, "top_k": 5, "nocache": True}

    resp = None
    for attempt in range(max_retries):
        resp = await client.post(url, json=payload)
        if resp.status_code == 429:
            wait = min(2**attempt * 2, 60)
            logger.warning(
                "Groq 429 on query %s, retry attempt %d after %ds",
                query.id, attempt + 1, wait,
            )
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()

    raise httpx.HTTPStatusError(
        f"Rate limited after {max_retries} retries",
        request=resp.request if resp else None,
        response=resp,
    )


async def collect_results(
    endpoint: str = "http://localhost:8000",
    dataset_path: str = "eval/synthetic_queries.json",
    limit: int | None = None,
) -> list[dict]:
    """Query the backend for N synthetic queries (or all), with batching."""
    queries = load_synthetic_queries(dataset_path)
    if limit is not None:
        queries = queries[:limit]

    logger.info("Collecting results for %d queries from %s", len(queries), endpoint)

    rows: list[dict] = []

    async with httpx.AsyncClient(timeout=120) as client:
        for i in range(0, len(queries), BATCH_SIZE):
            batch = queries[i : i + BATCH_SIZE]
            for q in batch:
                try:
                    data = await _query_endpoint(client, endpoint, q)
                except Exception as e:
                    logger.error("Failed query %s: %s", q.id, e)
                    data = {"answer": "", "sources": [], "confidence": {"score": 0}}

                # trust_score lives under `confidence.score` in v0.2+ backend,
                # but fall back to top-level `trust_score` for compatibility with
                # older mocks + potential schema drift.
                trust_score = None
                if isinstance(data.get("confidence"), dict):
                    trust_score = data["confidence"].get("score")
                if trust_score is None:
                    trust_score = data.get("trust_score")

                rows.append({
                    "question": q.text,
                    "answer": data.get("answer", ""),
                    "contexts": [s.get("text", s.get("content", "")) for s in data.get("sources", [])],
                    "ground_truth": q.expected_answer_substring or "",
                    "trust_score": trust_score,
                    "category": q.category,
                    "ground_truth_chunk_ids": q.ground_truth_chunk_ids,
                    "retrieved_chunk_ids": [
                        s.get("chunk_id", s.get("id", ""))
                        for s in data.get("sources", [])
                    ],
                })

            # Pause between batches to respect Groq rate limits (SIGN-104)
            if i + BATCH_SIZE < len(queries):
                logger.debug("Pausing %ds between batches", BATCH_PAUSE_SECONDS)
                await asyncio.sleep(BATCH_PAUSE_SECONDS)

    return rows


def compute_hit_at_5(rows: list[dict]) -> dict[str, float]:
    """Compute Hit@5 overall and by category. Deterministic (no LLM)."""
    hit_by_cat: dict[str, float] = {}
    for cat in ["semantic", "keyword", "hybrid"]:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        hits = [
            any(gt in r["retrieved_chunk_ids"] for gt in r["ground_truth_chunk_ids"])
            for r in cat_rows
        ]
        hit_by_cat[cat] = sum(hits) / len(hits)

    overall_hits = [
        any(gt in r["retrieved_chunk_ids"] for gt in r["ground_truth_chunk_ids"])
        for r in rows
    ]
    hit_by_cat["overall"] = sum(overall_hits) / len(overall_hits) if overall_hits else 0.0

    return hit_by_cat


# ---------------------------------------------------------------------------
# RAGAS evaluation driven by Gemini judge
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    rows: list[dict],
    batch_size: int = 5,
    judge_provider: str = "groq",
) -> dict[str, Any]:
    """Run RAGAS metrics on collected results.

    judge_provider choices:
      - "groq"   (default, v2-completion): Groq Llama 3.1 8B Instant
                 (500K TPD, fast inference, binary verdicts are OK on 8B)
      - "gemini": Gemini 2.5 Flash Lite (slow due to AFC internal retries,
                  kept as fallback)
      - "default": no judge override, RAGAS tries its default (OpenAI —
                   needs OPENAI_API_KEY; avoid unless you have one)

    Embeddings (for answer_relevancy) always use Gemini since Groq has
    no embedding API.
    """
    # Filter out rows with empty answer (failed queries) — RAGAS would crash
    valid_rows = [r for r in rows if r["answer"]]
    if len(valid_rows) < len(rows):
        logger.warning("Skipping %d rows with empty answer in RAGAS eval", len(rows) - len(valid_rows))

    if not valid_rows:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_query": [],
        }

    ragas_ds = Dataset.from_list([
        {k: r[k] for k in ["question", "answer", "contexts", "ground_truth"]}
        for r in valid_rows
    ])

    # RunConfig: Groq handles parallel well (500+ tok/s), Gemini needs serial
    # to survive free-tier RPM burst limits.
    if judge_provider == "groq":
        run_config = RunConfig(max_retries=5, max_wait=30, max_workers=4)
    else:
        run_config = RunConfig(max_retries=8, max_wait=90, max_workers=1)

    eval_kwargs = dict(
        dataset=ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        run_config=run_config,
        batch_size=batch_size,
    )
    if judge_provider == "groq":
        eval_kwargs["llm"] = _get_groq_judge()
        eval_kwargs["embeddings"] = _get_gemini_embeddings()
    elif judge_provider == "gemini":
        eval_kwargs["llm"] = _get_gemini_judge()
        eval_kwargs["embeddings"] = _get_gemini_embeddings()
    # else: judge_provider == "default" — let RAGAS use its OpenAI default

    result = evaluate(**eval_kwargs)

    df = result.to_pandas()
    return {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "context_recall": float(df["context_recall"].mean()),
        "per_query": df.to_dict(orient="records"),
    }


async def run_full_benchmark(
    endpoint: str = "http://localhost:8000",
    dataset_path: str = "eval/synthetic_queries.json",
    limit: int | None = None,
    mode: str = "hybrid",
    judge_provider: str = "groq",
) -> dict[str, Any]:
    """Run the complete benchmark: query backend, compute RAGAS + trust + hit@5.

    Args:
        endpoint: TrustRAG backend URL (should have HYBRID_ENABLED flipped to
            match `mode` before calling — this function does not flip the flag)
        dataset_path: path to synthetic_queries.json
        limit: number of queries to run (None = all 30)
        mode: "semantic" or "hybrid" (informational; toggle Railway HYBRID_ENABLED separately)
        judge_provider: "groq" (default, fast) / "gemini" (slow fallback) / "default" (OpenAI)
    """
    start_ts = time.time()
    rows = await collect_results(endpoint, dataset_path, limit=limit)

    # RAGAS metrics (synchronous internally)
    judge_label = {
        "groq": "groq-llama-3.1-8b-instant",
        "gemini": "gemini-2.5-flash-lite",
        "default": "ragas-default-openai",
    }.get(judge_provider, judge_provider)
    logger.info("Running RAGAS on %d rows (judge=%s)", len(rows), judge_label)
    ragas_results = run_ragas_evaluation(rows, judge_provider=judge_provider)

    # Trust distribution (deterministic, no LLM)
    trust_dist = compute_trust_metrics(rows)

    # Hit@5 (deterministic, no LLM)
    hit_at_5 = compute_hit_at_5(rows)

    elapsed = time.time() - start_ts

    return {
        "metadata": {
            "date": datetime.utcnow().isoformat() + "Z",
            "mode": mode,
            "endpoint": endpoint,
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "judge_model": judge_label,
            "queries_count": len(rows),
            "elapsed_seconds": round(elapsed, 1),
        },
        "ragas_summary": {
            k: v for k, v in ragas_results.items() if k != "per_query"
        },
        "ragas_per_query": ragas_results["per_query"],
        "trust_distribution": dict(trust_dist),
        "hit_at_5_by_category": hit_at_5,
        "hit_at_5_overall": hit_at_5.get("overall", 0.0),
        "per_query": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m trustrag_eval.ragas_pipeline",
        description="Run RAGAS benchmark against TrustRAG endpoint (Gemini-judged).",
    )
    ap.add_argument("--endpoint", required=True, help="Backend URL, e.g., https://trustrag-production.up.railway.app")
    ap.add_argument("--dataset", required=True, help="Path to synthetic_queries.json")
    ap.add_argument("--limit", type=int, default=None, help="Number of queries (default: all)")
    ap.add_argument("--mode", choices=["semantic", "hybrid"], required=True, help="Informational label (must match Railway HYBRID_ENABLED)")
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--judge", choices=["groq", "gemini", "default"], default="groq",
                    help="RAGAS judge provider: groq (default, Llama 8B fast), gemini (Flash Lite fallback, slow), default (OpenAI; needs OPENAI_API_KEY)")
    ap.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = asyncio.run(run_full_benchmark(
        endpoint=args.endpoint,
        dataset_path=args.dataset,
        limit=args.limit,
        mode=args.mode,
        judge_provider=args.judge,
    ))

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    # Print summary to stdout for easy piping
    summary = {
        "metadata": result["metadata"],
        "ragas_summary": result["ragas_summary"],
        "hit_at_5": {
            "overall": result["hit_at_5_overall"],
            **result["hit_at_5_by_category"],
        },
    }
    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(_main())
