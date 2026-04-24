"""Smoke tests for Gemini 2.0 Flash judge wrapper.

Gated on GOOGLE_API_KEY env var — skipped in CI without key.
Each test makes at most a few Gemini API calls (well within free tier).
"""

import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set; skipping Gemini smoke tests",
)


def test_gemini_judge_loads_without_error():
    from trustrag_eval.ragas_pipeline import _get_gemini_judge

    judge = _get_gemini_judge()
    assert judge is not None


def test_gemini_embeddings_load_without_error():
    from trustrag_eval.ragas_pipeline import _get_gemini_embeddings

    emb = _get_gemini_embeddings()
    assert emb is not None


def test_gemini_judge_missing_api_key_raises():
    """Temporarily unset env var to verify error message."""
    from trustrag_eval.ragas_pipeline import _get_gemini_judge

    saved = os.environ.pop("GOOGLE_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY not set"):
            _get_gemini_judge()
    finally:
        if saved is not None:
            os.environ["GOOGLE_API_KEY"] = saved


@pytest.mark.skipif(
    not (os.getenv("GROQ_API_KEY") and os.getenv("GOOGLE_API_KEY")),
    reason="Needs both GROQ_API_KEY (judge) and GOOGLE_API_KEY (embeddings)",
)
def test_groq_judge_scores_one_sample():
    """End-to-end: Groq 8B judge + Gemini embeddings compute RAGAS on trivial example.

    Goes through run_ragas_evaluation (the real entry point). ~4s runtime.
    Replaces the former Gemini-only end-to-end test which took 30s+ per metric
    due to AFC internal retries (gemini-2.5-flash-lite still too slow for CI).
    """
    from trustrag_eval.ragas_pipeline import run_ragas_evaluation

    rows = [{
        "question": "What is 2+2?",
        "answer": "2+2 equals 4.",
        "contexts": ["Basic arithmetic: 2+2 = 4."],
        "ground_truth": "4",
        "trust_score": 80,
        "category": "semantic",
        "ground_truth_chunk_ids": [],
        "retrieved_chunk_ids": [],
    }]

    result = run_ragas_evaluation(rows, judge_provider="groq")
    # Obviously correct trivial example: faithfulness should be non-zero
    assert result["faithfulness"] >= 0.3, f"Expected faithfulness >= 0.3, got {result['faithfulness']}"
    import math
    assert not math.isnan(result["faithfulness"]), "faithfulness was NaN"
