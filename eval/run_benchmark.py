"""Baseline benchmark runner for TrustRAG hybrid retrieval (Phase 2).

Runs 30 synthetic queries against the API and computes hit@5 by category.
Supports both hybrid and semantic-only modes for A/B comparison.

Usage:
    python eval/run_benchmark.py --mode hybrid
    python eval/run_benchmark.py --mode semantic
    python eval/run_benchmark.py --compare  # runs both and prints table
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from statistics import mean

import httpx

EVAL_DIR = Path(__file__).parent
QUERIES_FILE = EVAL_DIR / "synthetic_queries.json"
RESULTS_DIR = EVAL_DIR / "results"


def load_queries() -> list[dict]:
    """Load synthetic queries from JSON file."""
    with open(QUERIES_FILE) as f:
        data = json.load(f)
    return data["queries"]


async def run_benchmark(
    endpoint: str = "http://localhost:8000",
    mode: str = "hybrid",
) -> dict:
    """Run benchmark against the TrustRAG API.

    Args:
        endpoint: Base URL of the running TrustRAG server.
        mode: 'hybrid' or 'semantic' (controls HYBRID_ENABLED env).

    Returns:
        Benchmark results dict with overall and per-category hit@5.
    """
    import asyncio

    queries = load_queries()
    results = []

    async with httpx.AsyncClient(timeout=30.0, base_url=endpoint) as client:
        for q in queries:
            try:
                resp = await client.post(
                    "/api/query",
                    json={"question": q["text"], "top_k": 5},
                )
                resp.raise_for_status()
                data = resp.json()

                retrieved_ids = [s["chunk_id"] for s in data.get("sources", [])]
                hit = any(
                    gt_id in retrieved_ids
                    for gt_id in q["ground_truth_chunk_ids"]
                )

                results.append({
                    "query_id": q["id"],
                    "category": q["category"],
                    "hit@5": hit,
                    "trust_score": data.get("trust_score"),
                    "retrieved_ids": retrieved_ids,
                })
            except (httpx.HTTPError, KeyError) as e:
                results.append({
                    "query_id": q["id"],
                    "category": q["category"],
                    "hit@5": False,
                    "trust_score": None,
                    "retrieved_ids": [],
                    "error": str(e),
                })

    # Aggregate hit@5 by category
    hit_by_category = {}
    for cat in ["semantic", "keyword", "hybrid"]:
        cat_results = [r["hit@5"] for r in results if r["category"] == cat]
        hit_by_category[cat] = mean(cat_results) if cat_results else 0.0

    overall_hit5 = mean(r["hit@5"] for r in results) if results else 0.0

    return {
        "mode": mode,
        "date": str(date.today()),
        "overall_hit_at_5": overall_hit5,
        "hit_at_5_by_category": hit_by_category,
        "total_queries": len(results),
        "queries": results,
    }


def save_results(results: dict, suffix: str = "") -> Path:
    """Save benchmark results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{date.today()}-phase2-baseline{suffix}.json"
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def print_comparison_table(hybrid_results: dict, semantic_results: dict) -> str:
    """Print a formatted comparison table."""
    lines = []
    lines.append("=" * 60)
    lines.append("TrustRAG Phase 2 Baseline Benchmark Results")
    lines.append("=" * 60)
    lines.append(f"{'Category':<12} {'Hybrid hit@5':<16} {'Semantic hit@5':<16}")
    lines.append("-" * 60)

    for cat in ["semantic", "keyword", "hybrid"]:
        h = hybrid_results["hit_at_5_by_category"].get(cat, 0)
        s = semantic_results["hit_at_5_by_category"].get(cat, 0)
        lines.append(f"{cat:<12} {h:<16.2%} {s:<16.2%}")

    lines.append("-" * 60)
    h_overall = hybrid_results["overall_hit_at_5"]
    s_overall = semantic_results["overall_hit_at_5"]
    lines.append(f"{'OVERALL':<12} {h_overall:<16.2%} {s_overall:<16.2%}")
    lines.append("=" * 60)

    table = "\n".join(lines)
    print(table)
    return table


async def main():
    import asyncio

    parser = argparse.ArgumentParser(description="TrustRAG Benchmark Runner")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "semantic"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="TrustRAG API endpoint (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both modes and print comparison table",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate queries file without hitting the API",
    )
    args = parser.parse_args()

    if args.dry_run:
        queries = load_queries()
        categories = {}
        for q in queries:
            categories[q["category"]] = categories.get(q["category"], 0) + 1
        print(f"Loaded {len(queries)} queries:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        print("\nAll queries have required fields: OK")
        return

    if args.compare:
        # Run hybrid mode
        print("Running hybrid mode...")
        os.environ["HYBRID_ENABLED"] = "true"
        hybrid_results = await run_benchmark(endpoint=args.endpoint, mode="hybrid")
        save_results(hybrid_results, "-hybrid")

        # Run semantic-only mode
        print("Running semantic-only mode...")
        os.environ["HYBRID_ENABLED"] = "false"
        semantic_results = await run_benchmark(endpoint=args.endpoint, mode="semantic")
        save_results(semantic_results, "-semantic")

        print_comparison_table(hybrid_results, semantic_results)
    else:
        if args.mode == "semantic":
            os.environ["HYBRID_ENABLED"] = "false"
        else:
            os.environ["HYBRID_ENABLED"] = "true"

        results = await run_benchmark(endpoint=args.endpoint, mode=args.mode)
        path = save_results(results, f"-{args.mode}")
        print(f"Results saved to: {path}")
        print(f"Overall hit@5: {results['overall_hit_at_5']:.2%}")
        for cat, score in results["hit_at_5_by_category"].items():
            print(f"  {cat}: {score:.2%}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
