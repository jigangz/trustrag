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


def test_gemini_judge_actually_scores_one_sample():
    """End-to-end: can Gemini judge compute RAGAS metrics on a trivial example?

    Goes through run_ragas_evaluation (the real entry point) so this covers
    the legacy-underscored-metrics + LangchainLLMWrapper path we actually use.
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

    result = run_ragas_evaluation(rows, use_gemini=True)
    # Obviously correct: faithfulness should be non-zero for a trivial match
    # (Gemini may score anywhere in [0.3, 1.0] depending on its noncommittal rating)
    assert result["faithfulness"] >= 0.3, f"Expected faithfulness >= 0.3, got {result['faithfulness']}"
    # And it should not be NaN
    import math
    assert not math.isnan(result["faithfulness"]), "faithfulness was NaN"
