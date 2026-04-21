"""Real-data benchmark runner for TrustRAG.

Runs synthetic queries against the LIVE backend. Computes real metrics
from the actual API response shape:
- Trust score distribution (from data.confidence.score)
- Average retrieval similarity (from data.sources[].similarity)
- Answer substring match rate (expected_answer_substring in data.answer)

Usage:
    # Hybrid mode (current .env config, defaults HYBRID_ENABLED=true)
    python eval/run_real_benchmark.py --mode hybrid --out eval/results/real-hybrid.json

    # Semantic-only mode (requires HYBRID_ENABLED=false + backend restart)
    python eval/run_real_benchmark.py --mode semantic --out eval/results/real-semantic.json

    # Compare
    python eval/run_real_benchmark.py --compare real-hybrid.json real-semantic.json
"""

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from statistics import mean, median

import httpx

EVAL_DIR = Path(__file__).parent
QUERIES_FILE = EVAL_DIR / "synthetic_queries.json"
RESULTS_DIR = EVAL_DIR / "results"


def load_queries() -> list[dict]:
    with open(QUERIES_FILE) as f:
        return json.load(f)["queries"]


async def query_one(
    client: httpx.AsyncClient, endpoint: str, q: dict, max_retries: int = 3
) -> dict:
    """Query one question; return measurable fields. Handles 429 with backoff."""
    url = f"{endpoint}/api/query/"
    payload = {"question": q["text"], "top_k": 5}

    for attempt in range(max_retries):
        try:
            resp = await client.post(url, json=payload, follow_redirects=True)
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"  [{q['id']}] 429 rate-limited, backing off {wait}s")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            # Extract fields from ACTUAL API shape
            confidence = data.get("confidence", {})
            trust_score = confidence.get("score")
            sources = data.get("sources", [])
            top_similarity = sources[0]["similarity"] if sources else 0.0
            answer = data.get("answer", "")

            # Check if expected answer substring is in the response
            expected = (q.get("expected_answer_substring") or "").lower()
            substring_match = expected in answer.lower() if expected else None

            return {
                "query_id": q["id"],
                "category": q["category"],
                "trust_score": trust_score,
                "trust_level": confidence.get("level"),
                "top_similarity": top_similarity,
                "answer_length": len(answer),
                "num_sources": len(sources),
                "substring_match": substring_match,
                "expected_substring": expected,
                "hallucination_passed": data.get("hallucination_check", {}).get("passed"),
                "audit_id": data.get("audit_id"),
                "error": None,
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "query_id": q["id"],
                    "category": q["category"],
                    "trust_score": None,
                    "top_similarity": 0.0,
                    "substring_match": False,
                    "error": f"{type(e).__name__}: {e}",
                }
            await asyncio.sleep(2)


async def run_benchmark(
    endpoint: str, mode: str, sample: int | None = None, pause_ms: int = 500
) -> dict:
    queries = load_queries()
    if sample:
        queries = queries[:sample]

    print(f"Running {len(queries)} queries in '{mode}' mode against {endpoint}")

    async with httpx.AsyncClient(timeout=90.0) as client:
        results = []
        for i, q in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {q['id']} ({q['category']}): {q['text'][:60]}...")
            r = await query_one(client, endpoint, q)
            results.append(r)
            if pause_ms:
                await asyncio.sleep(pause_ms / 1000)

    # Aggregate metrics
    valid = [r for r in results if r.get("trust_score") is not None]
    if not valid:
        return {"mode": mode, "error": "No valid results", "queries": results}

    trust_scores = [r["trust_score"] for r in valid]
    similarities = [r["top_similarity"] for r in valid if r["top_similarity"] > 0]
    substring_matches = [r["substring_match"] for r in valid if r["substring_match"] is not None]
    hallucination_passed = [r["hallucination_passed"] for r in valid if r["hallucination_passed"] is not None]

    by_category = {}
    for cat in ["semantic", "keyword", "hybrid"]:
        cat_valid = [r for r in valid if r["category"] == cat]
        if not cat_valid:
            continue
        by_category[cat] = {
            "count": len(cat_valid),
            "trust_mean": mean(r["trust_score"] for r in cat_valid),
            "trust_median": median(r["trust_score"] for r in cat_valid),
            "top_similarity_mean": mean(r["top_similarity"] for r in cat_valid if r["top_similarity"] > 0) if any(r["top_similarity"] > 0 for r in cat_valid) else 0,
            "substring_match_rate": (
                sum(1 for r in cat_valid if r.get("substring_match")) / len(cat_valid)
                if any(r.get("substring_match") is not None for r in cat_valid) else None
            ),
        }

    return {
        "mode": mode,
        "date": str(date.today()),
        "total_queries": len(results),
        "valid_responses": len(valid),
        "errors": sum(1 for r in results if r.get("error")),
        "trust_distribution": {
            "mean": mean(trust_scores),
            "median": median(trust_scores),
            "min": min(trust_scores),
            "max": max(trust_scores),
            "flagged_pct": sum(1 for s in trust_scores if s < 50) / len(trust_scores),
        },
        "retrieval": {
            "top_similarity_mean": mean(similarities) if similarities else 0,
            "top_similarity_median": median(similarities) if similarities else 0,
        },
        "answer_quality": {
            "substring_match_rate": (
                sum(1 for m in substring_matches if m) / len(substring_matches)
                if substring_matches else None
            ),
            "hallucination_pass_rate": (
                sum(1 for p in hallucination_passed if p) / len(hallucination_passed)
                if hallucination_passed else None
            ),
        },
        "by_category": by_category,
        "per_query": results,
    }


def generate_comparison(hybrid_path: str, semantic_path: str, out_path: str) -> str:
    with open(hybrid_path) as f:
        hyb = json.load(f)
    with open(semantic_path) as f:
        sem = json.load(f)

    def fmt(v, pct=False):
        if v is None:
            return "n/a"
        return f"{v * 100:.0f}%" if pct else f"{v:.2f}" if isinstance(v, float) else str(v)

    def delta(h, s, pct=False):
        if h is None or s is None:
            return "n/a"
        d = h - s
        sign = "+" if d > 0 else ""
        return f"**{sign}{d * 100:.0f}pp**" if pct else f"{sign}{d:.2f}"

    lines = [
        "# TrustRAG Real-Data Benchmark: Semantic-only vs Hybrid (RRF)",
        "",
        f"**Corpus**: OSHA 3150 (A Guide to Scaffold Use in the Construction Industry, 174 chunks)  ",
        f"**Queries**: {hyb['total_queries']} synthetic questions across 3 categories  ",
        f"**Date**: {date.today()}  ",
        f"**Backend**: llama-3.3-70b-versatile on Groq",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Semantic-only | Hybrid (RRF k=60) | Delta |",
        "|--------|---------------|-------------------|-------|",
        f"| Trust score (mean) | {fmt(sem['trust_distribution']['mean'])} | {fmt(hyb['trust_distribution']['mean'])} | {delta(hyb['trust_distribution']['mean'], sem['trust_distribution']['mean'])} |",
        f"| Trust score (median) | {fmt(sem['trust_distribution']['median'])} | {fmt(hyb['trust_distribution']['median'])} | {delta(hyb['trust_distribution']['median'], sem['trust_distribution']['median'])} |",
        f"| Flagged rate (<50) | {fmt(sem['trust_distribution']['flagged_pct'], pct=True)} | {fmt(hyb['trust_distribution']['flagged_pct'], pct=True)} | {delta(hyb['trust_distribution']['flagged_pct'], sem['trust_distribution']['flagged_pct'], pct=True)} |",
        f"| Top-1 similarity (mean) | {fmt(sem['retrieval']['top_similarity_mean'])} | {fmt(hyb['retrieval']['top_similarity_mean'])} | {delta(hyb['retrieval']['top_similarity_mean'], sem['retrieval']['top_similarity_mean'])} |",
        f"| Substring match rate | {fmt(sem['answer_quality']['substring_match_rate'], pct=True)} | {fmt(hyb['answer_quality']['substring_match_rate'], pct=True)} | {delta(hyb['answer_quality']['substring_match_rate'], sem['answer_quality']['substring_match_rate'], pct=True)} |",
        f"| Hallucination pass rate | {fmt(sem['answer_quality']['hallucination_pass_rate'], pct=True)} | {fmt(hyb['answer_quality']['hallucination_pass_rate'], pct=True)} | {delta(hyb['answer_quality']['hallucination_pass_rate'], sem['answer_quality']['hallucination_pass_rate'], pct=True)} |",
        "",
        "## By Category (Trust mean)",
        "",
        "| Category | Semantic | Hybrid | Delta |",
        "|----------|----------|--------|-------|",
    ]
    for cat in ["semantic", "keyword", "hybrid"]:
        s = sem.get("by_category", {}).get(cat, {}).get("trust_mean")
        h = hyb.get("by_category", {}).get(cat, {}).get("trust_mean")
        lines.append(f"| {cat} | {fmt(s)} | {fmt(h)} | {delta(h, s)} |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- **Corpus**: Single OSHA scaffold safety document (174 chunks). All queries evaluated against this corpus.",
        "- **Substring match**: Case-insensitive match of `expected_answer_substring` in the generated answer. Coarse proxy for factual correctness.",
        "- **Hallucination pass rate**: Fraction of answers where the secondary-LLM hallucination check flagged no issues.",
        "- **Hit@5 NOT measured**: Ground-truth chunk IDs in the synthetic dataset are placeholders (c_042) and have not been remapped to the real corpus. A larger multi-document corpus is required for meaningful hit@5.",
        "- These are real measurements, not projections.",
        "",
    ])

    report = "\n".join(lines)
    Path(out_path).write_text(report, encoding="utf-8")
    return report


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--mode", choices=["hybrid", "semantic"], default="hybrid")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sample", type=int, default=None, help="Limit to first N queries")
    parser.add_argument("--pause-ms", type=int, default=500)
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("HYBRID_JSON", "SEMANTIC_JSON"),
    )
    args = parser.parse_args()

    if args.compare:
        out = RESULTS_DIR / f"{date.today()}-real-comparison.md"
        report = generate_comparison(args.compare[0], args.compare[1], str(out))
        print(report)
        print(f"\nSaved to: {out}")
        return

    results = await run_benchmark(
        endpoint=args.endpoint,
        mode=args.mode,
        sample=args.sample,
        pause_ms=args.pause_ms,
    )

    out = args.out or RESULTS_DIR / f"{date.today()}-real-{args.mode}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Summary ({args.mode}) ===")
    print(f"  Valid responses: {results['valid_responses']}/{results['total_queries']}")
    print(f"  Trust mean: {results['trust_distribution']['mean']:.1f}")
    print(f"  Trust flagged %: {results['trust_distribution']['flagged_pct'] * 100:.0f}%")
    print(f"  Top-1 similarity mean: {results['retrieval']['top_similarity_mean']:.3f}")
    if results['answer_quality']['substring_match_rate'] is not None:
        print(f"  Substring match rate: {results['answer_quality']['substring_match_rate'] * 100:.0f}%")
    if results['answer_quality']['hallucination_pass_rate'] is not None:
        print(f"  Hallucination pass rate: {results['answer_quality']['hallucination_pass_rate'] * 100:.0f}%")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
