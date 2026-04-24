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
    """End-to-end: can Gemini judge compute faithfulness on a trivial example?"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics.collections import faithfulness

    from trustrag_eval.ragas_pipeline import _get_gemini_judge, _get_gemini_embeddings

    sample = Dataset.from_list([
        {
            "question": "What is 2+2?",
            "answer": "2+2 equals 4.",
            "contexts": ["Basic arithmetic: 2+2 = 4."],
            "ground_truth": "4",
        }
    ])

    result = evaluate(
        dataset=sample,
        metrics=[faithfulness],
        llm=_get_gemini_judge(),
        embeddings=_get_gemini_embeddings(),
    )

    df = result.to_pandas()
    score = float(df["faithfulness"].iloc[0])
    # Obviously correct: faithfulness should be high
    assert 0.5 <= score <= 1.0, f"Expected faithfulness in [0.5, 1.0], got {score}"
