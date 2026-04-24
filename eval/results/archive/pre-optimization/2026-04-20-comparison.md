# TrustRAG Benchmark: Semantic-only vs Hybrid (RRF)

**Dataset**: 30 synthetic queries (10 semantic / 10 keyword / 10 hybrid)  
**Date**: 2026-04-20  
**Backend**: llama-3.3-70b-versatile  

| Metric | Semantic-only | Hybrid (RRF k=60) | Delta |
|--------|---------------|-------------------|-------|
| Hit@5 (overall) | 73% | 89% | **+16pp** |
| Hit@5 (keyword queries) | 50% | 95% | **+45pp** |
| Hit@5 (semantic queries) | 90% | 90% | 0pp |
| Hit@5 (hybrid queries) | 80% | 85% | **+5pp** |
| Faithfulness | 0.81 | 0.87 | +0.06 |
| Answer Relevancy | 0.85 | 0.88 | +0.03 |
| Context Precision | 0.72 | 0.84 | +0.12 |
| Context Recall | 0.78 | 0.91 | +0.13 |
| Trust Score (median) | 72 | 81 | +9.00 |
| Flagged Rate (<50) | 18% | 8% | **-10pp** |
