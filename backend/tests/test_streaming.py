"""Tests for streaming.QueryTask — embedding reuse optimization."""

import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock fastembed before importing modules that depend on it
sys.modules.setdefault("fastembed", MagicMock())


@pytest.mark.asyncio
async def test_verify_trust_reuses_query_embedding_from_retrieve():
    """QueryTask._verify_trust must not re-embed; use _retrieve's cached embedding."""
    from services.streaming import QueryTask

    task = QueryTask(query_id="test-01", text="test question", top_k=5)

    # Simulate _retrieve having stored the embedding
    task._query_embedding = [0.1] * 384

    mock_trust = MagicMock()
    mock_trust.score = 85
    mock_trust.level = "high"
    mock_trust.retrieval_similarity = 0.9
    mock_trust.source_count_score = 20
    mock_trust.source_agreement = 0.85
    mock_trust.hallucination_free = True
    mock_trust.hallucination_flags = []

    with patch("services.trust_verifier.compute_trust_score", new=AsyncMock(return_value=mock_trust)) as mock_trust_fn:
        await task._verify_trust("test answer", [{"content": "src"}])

    mock_trust_fn.assert_called_once()
    # Verify the embedding passed in was the cached one, not a fresh embed call
    args, kwargs = mock_trust_fn.call_args
    passed_embedding = args[2] if len(args) >= 3 else kwargs.get("query_embedding")
    assert passed_embedding == [0.1] * 384
