# trustrag-eval

RAGAS-based evaluation pipeline with trust-specific metrics for TrustRAG.

## Getting Started

```bash
# Install in development mode
pip install -e packages/trustrag-eval

# Run tests
cd packages/trustrag-eval && pytest tests/ -v
```

## Metrics Explained

### RAGAS Metrics (industry-standard)

| Metric | What it measures |
|--------|-----------------|
| Faithfulness | Does the answer stay within retrieved context? |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | How relevant are the retrieved chunks? |
| Context Recall | Did we retrieve the chunks needed for ground truth? |

### Trust-Specific Metrics (TrustRAG)

| Metric | What it measures |
|--------|-----------------|
| Trust Score Distribution (p25/p50/p75/mean) | Overall trust score health across queries |
| Flagged Rate | Percentage of queries with trust_score < 50 |
| Hit@5 | Was the ground-truth chunk in the top-5 retrieved? |
| Hit@5 by Category | Hit rate broken down by query type (semantic/keyword/hybrid) |

## Usage

```python
from trustrag_eval import load_synthetic_queries, compute_trust_metrics

# Load benchmark dataset
queries = load_synthetic_queries("eval/synthetic_queries.json")

# Compute trust distribution from results
results = [{"trust_score": 85}, {"trust_score": 72}, ...]
dist = compute_trust_metrics(results)
print(f"Median trust: {dist['p50']}, Flagged: {dist['flagged_pct']:.0%}")
```
