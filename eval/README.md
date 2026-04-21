# TrustRAG Evaluation Suite

## Overview

This directory contains the synthetic evaluation dataset and benchmark runner for measuring TrustRAG retrieval quality.

## Files

- `synthetic_queries.json` — 30 queries (10 semantic / 10 keyword / 10 hybrid)
- `run_benchmark.py` — Benchmark runner that computes hit@5 by category
- `results/` — Saved benchmark results (JSON)

## Running the Benchmark

### Prerequisites

- TrustRAG backend running locally on port 8000
- Documents ingested (the queries reference OSHA safety documents)

### Commands

```bash
# Dry run: validate query file without hitting API
python eval/run_benchmark.py --dry-run

# Run hybrid mode
python eval/run_benchmark.py --mode hybrid

# Run semantic-only mode
python eval/run_benchmark.py --mode semantic

# Compare both modes side-by-side
python eval/run_benchmark.py --compare

# Custom endpoint
python eval/run_benchmark.py --endpoint http://localhost:9000 --mode hybrid
```

### Output

Results are saved to `eval/results/YYYY-MM-DD-phase2-baseline-{mode}.json` with this structure:

```json
{
  "mode": "hybrid",
  "date": "2026-04-20",
  "overall_hit_at_5": 0.73,
  "hit_at_5_by_category": {
    "semantic": 0.80,
    "keyword": 0.70,
    "hybrid": 0.70
  },
  "total_queries": 30,
  "queries": [...]
}
```

## Adding Queries

Add entries to `synthetic_queries.json` with this schema:

```json
{
  "id": "Q031",
  "text": "Your query text here",
  "category": "semantic|keyword|hybrid",
  "ground_truth_chunk_ids": ["chunk_id_1", "chunk_id_2"],
  "expected_answer_substring": "expected text in answer"
}
```

### Category Guidelines

- **semantic**: Natural language questions testing conceptual understanding
- **keyword**: Exact terms, regulation numbers, or technical identifiers
- **hybrid**: Mix of natural language with specific technical terms

## Baseline Results (Phase 2)

| Category | Hybrid hit@5 | Semantic hit@5 |
|----------|-------------|----------------|
| semantic | TBD | TBD |
| keyword | TBD | TBD |
| hybrid | TBD | TBD |
| **OVERALL** | **TBD** | **TBD** |

*Results will be populated after running against ingested OSHA documents.*
