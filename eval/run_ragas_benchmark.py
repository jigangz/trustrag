"""Full RAGAS benchmark runner for TrustRAG.

Runs 30 synthetic queries against the backend, computes:
- 4 RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall)
- Trust score distribution (p25/p50/p75/mean/flagged_pct)
- Hit@5 overall and by category (semantic/keyword/hybrid)

Usage:
    # Run with hybrid retrieval (default)
    python eval/run_ragas_benchmark.py --out eval/results/2026-04-20-ragas-hybrid.json

    # Run with semantic-only retrieval
    python eval/run_ragas_benchmark.py --out eval/results/2026-04-20-ragas-semantic.json

    # Generate comparison report from two result files
    python eval/run_ragas_benchmark.py --compare \
        eval/results/2026-04-20-ragas-semantic.json \
        eval/results/2026-04-20-ragas-hybrid.json
"""

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

# Add packages to path for dev mode
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "trustrag-eval" / "src"))

from trustrag_eval.ragas_pipeline import run_full_benchmark


def generate_comparison(semantic_path: str, hybrid_path: str, output_path: str) -> str:
    """Generate a comparison markdown from two result JSON files."""
    with open(semantic_path) as f:
        sem = json.load(f)
    with open(hybrid_path) as f:
        hyb = json.load(f)

    def fmt_pct(v: float) -> str:
        return f"{v * 100:.0f}%" if v <= 1 else f"{v:.0f}%"

    def fmt_score(v: float) -> str:
        return f"{v:.2f}"

    def delta(h: float, s: float, is_pct: bool = False) -> str:
        d = h - s
        if is_pct:
            sign = "+" if d > 0 else ""
            return f"**{sign}{d * 100:.0f}pp**" if abs(d) > 0.01 else "0pp"
        sign = "+" if d > 0 else ""
        return f"{sign}{d:.2f}"

    sem_hit = sem.get("hit_at_5_by_category", {})
    hyb_hit = hyb.get("hit_at_5_by_category", {})
    sem_ragas = sem.get("ragas_summary", {})
    hyb_ragas = hyb.get("ragas_summary", {})
    sem_trust = sem.get("trust_distribution", {})
    hyb_trust = hyb.get("trust_distribution", {})

    lines = [
        "# TrustRAG Benchmark: Semantic-only vs Hybrid (RRF)",
        "",
        f"**Dataset**: 30 synthetic queries (10 semantic / 10 keyword / 10 hybrid)  ",
        f"**Date**: {date.today()}  ",
        "**Backend**: llama-3.3-70b-versatile  ",
        "",
        "| Metric | Semantic-only | Hybrid (RRF k=60) | Delta |",
        "|--------|---------------|-------------------|-------|",
        f"| Hit@5 (overall) | {fmt_pct(sem.get('hit_at_5_overall', 0))} | {fmt_pct(hyb.get('hit_at_5_overall', 0))} | {delta(hyb.get('hit_at_5_overall', 0), sem.get('hit_at_5_overall', 0), True)} |",
        f"| Hit@5 (keyword queries) | {fmt_pct(sem_hit.get('keyword', 0))} | {fmt_pct(hyb_hit.get('keyword', 0))} | {delta(hyb_hit.get('keyword', 0), sem_hit.get('keyword', 0), True)} |",
        f"| Hit@5 (semantic queries) | {fmt_pct(sem_hit.get('semantic', 0))} | {fmt_pct(hyb_hit.get('semantic', 0))} | {delta(hyb_hit.get('semantic', 0), sem_hit.get('semantic', 0), True)} |",
        f"| Hit@5 (hybrid queries) | {fmt_pct(sem_hit.get('hybrid', 0))} | {fmt_pct(hyb_hit.get('hybrid', 0))} | {delta(hyb_hit.get('hybrid', 0), sem_hit.get('hybrid', 0), True)} |",
        f"| Faithfulness | {fmt_score(sem_ragas.get('faithfulness', 0))} | {fmt_score(hyb_ragas.get('faithfulness', 0))} | {delta(hyb_ragas.get('faithfulness', 0), sem_ragas.get('faithfulness', 0))} |",
        f"| Answer Relevancy | {fmt_score(sem_ragas.get('answer_relevancy', 0))} | {fmt_score(hyb_ragas.get('answer_relevancy', 0))} | {delta(hyb_ragas.get('answer_relevancy', 0), sem_ragas.get('answer_relevancy', 0))} |",
        f"| Context Precision | {fmt_score(sem_ragas.get('context_precision', 0))} | {fmt_score(hyb_ragas.get('context_precision', 0))} | {delta(hyb_ragas.get('context_precision', 0), sem_ragas.get('context_precision', 0))} |",
        f"| Context Recall | {fmt_score(sem_ragas.get('context_recall', 0))} | {fmt_score(hyb_ragas.get('context_recall', 0))} | {delta(hyb_ragas.get('context_recall', 0), sem_ragas.get('context_recall', 0))} |",
        f"| Trust Score (median) | {sem_trust.get('p50', 0):.0f} | {hyb_trust.get('p50', 0):.0f} | {delta(hyb_trust.get('p50', 0), sem_trust.get('p50', 0))} |",
        f"| Flagged Rate (<50) | {fmt_pct(sem_trust.get('flagged_pct', 0))} | {fmt_pct(hyb_trust.get('flagged_pct', 0))} | {delta(hyb_trust.get('flagged_pct', 0), sem_trust.get('flagged_pct', 0), True)} |",
        "",
    ]

    report = "\n".join(lines)
    Path(output_path).write_text(report, encoding="utf-8")
    return report


async def main():
    parser = argparse.ArgumentParser(description="TrustRAG RAGAS Benchmark Runner")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="TrustRAG API endpoint",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: eval/results/YYYY-MM-DD-ragas.json)",
    )
    parser.add_argument(
        "--dataset",
        default="eval/synthetic_queries.json",
        help="Path to synthetic queries JSON",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("SEMANTIC_JSON", "HYBRID_JSON"),
        help="Generate comparison markdown from two result files",
    )

    args = parser.parse_args()

    if args.compare:
        output_path = f"eval/results/{date.today()}-comparison.md"
        report = generate_comparison(args.compare[0], args.compare[1], output_path)
        print(report)
        print(f"\nSaved to: {output_path}")
        return

    output = args.out or f"eval/results/{date.today()}-ragas.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Running RAGAS benchmark against {args.endpoint}...")
    print(f"Dataset: {args.dataset}")

    results = await run_full_benchmark(
        endpoint=args.endpoint,
        dataset_path=args.dataset,
    )

    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output}")
    print(f"Total queries: {results['total_queries']}")
    print(f"Hit@5 overall: {results['hit_at_5_overall']:.2%}")
    print(f"Trust median: {results['trust_distribution']['p50']:.0f}")
    if results.get("ragas_summary"):
        print(f"Faithfulness: {results['ragas_summary']['faithfulness']:.2f}")
        print(f"Answer Relevancy: {results['ragas_summary']['answer_relevancy']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
