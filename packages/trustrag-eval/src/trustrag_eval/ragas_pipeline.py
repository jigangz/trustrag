"""RAGAS evaluation pipeline with rate-limit handling for TrustRAG benchmarks."""

import asyncio
import time
from typing import Any

import httpx
from datasets import Dataset
from ragas import RunConfig, evaluate
from ragas.metrics.collections import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from trustrag_eval.dataset import Query, load_synthetic_queries
from trustrag_eval.trust_metrics import compute_trust_metrics


BATCH_SIZE = 5
BATCH_PAUSE_SECONDS = 10


async def _query_endpoint(
    client: httpx.AsyncClient,
    endpoint: str,
    query: Query,
    max_retries: int = 5,
) -> dict:
    """Query the TrustRAG backend with exponential backoff on 429."""
    url = f"{endpoint}/api/query"
    payload = {"question": query.text, "top_k": 5}

    for attempt in range(max_retries):
        resp = await client.post(url, json=payload)
        if resp.status_code == 429:
            wait = min(2**attempt * 2, 60)
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()

    raise httpx.HTTPStatusError(
        f"Rate limited after {max_retries} retries",
        request=resp.request,
        response=resp,
    )


async def collect_results(
    endpoint: str = "http://localhost:8000",
    dataset_path: str = "eval/synthetic_queries.json",
) -> list[dict]:
    """Query the backend for all synthetic queries, with batching and backoff."""
    queries = load_synthetic_queries(dataset_path)
    rows: list[dict] = []

    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(queries), BATCH_SIZE):
            batch = queries[i : i + BATCH_SIZE]
            for q in batch:
                data = await _query_endpoint(client, endpoint, q)
                rows.append({
                    "question": q.text,
                    "answer": data.get("answer", ""),
                    "contexts": [s["content"] for s in data.get("sources", [])],
                    "ground_truth": q.expected_answer_substring or "",
                    "trust_score": data.get("trust_score"),
                    "category": q.category,
                    "ground_truth_chunk_ids": q.ground_truth_chunk_ids,
                    "retrieved_chunk_ids": [
                        s.get("chunk_id", s.get("id", ""))
                        for s in data.get("sources", [])
                    ],
                })

            # Pause between batches to respect Groq rate limits (SIGN-104)
            if i + BATCH_SIZE < len(queries):
                await asyncio.sleep(BATCH_PAUSE_SECONDS)

    return rows


def compute_hit_at_5(rows: list[dict]) -> dict[str, float]:
    """Compute Hit@5 overall and by category."""
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


def run_ragas_evaluation(rows: list[dict], batch_size: int = 5) -> dict[str, float]:
    """Run RAGAS metrics on collected results with retry/backoff via RunConfig."""
    ragas_ds = Dataset.from_list([
        {k: r[k] for k in ["question", "answer", "contexts", "ground_truth"]}
        for r in rows
    ])

    # RunConfig handles retries + exponential backoff for LLM calls (SIGN-104)
    run_config = RunConfig(
        max_retries=10,
        max_wait=60,
        max_workers=4,  # Limit concurrency to avoid Groq rate limits
    )

    result = evaluate(
        ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        run_config=run_config,
        batch_size=batch_size,
    )

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
) -> dict[str, Any]:
    """Run the complete benchmark: query backend, compute RAGAS + trust + hit@5."""
    rows = await collect_results(endpoint, dataset_path)

    # RAGAS metrics (synchronous, uses internal async)
    ragas_results = run_ragas_evaluation(rows)

    # Trust distribution
    trust_dist = compute_trust_metrics(rows)

    # Hit@5
    hit_at_5 = compute_hit_at_5(rows)

    return {
        "ragas_summary": {
            k: v for k, v in ragas_results.items() if k != "per_query"
        },
        "ragas_per_query": ragas_results["per_query"],
        "trust_distribution": dict(trust_dist),
        "hit_at_5_by_category": hit_at_5,
        "hit_at_5_overall": hit_at_5.get("overall", 0.0),
        "total_queries": len(rows),
    }
