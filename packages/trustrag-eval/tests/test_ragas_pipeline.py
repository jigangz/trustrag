"""Tests for RAGAS pipeline (mock endpoint, no real LLM calls)."""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from trustrag_eval.ragas_pipeline import (
    collect_results,
    compute_hit_at_5,
)


@pytest.fixture
def sample_rows():
    """Sample benchmark rows for testing aggregations."""
    return [
        {
            "question": "What is fall protection?",
            "answer": "6 feet threshold",
            "contexts": ["Fall protection is required at 6 feet"],
            "ground_truth": "6 feet",
            "trust_score": 85,
            "category": "semantic",
            "ground_truth_chunk_ids": ["c_042"],
            "retrieved_chunk_ids": ["c_042", "c_050", "c_051"],
        },
        {
            "question": "OSHA 1926.501",
            "answer": "Fall protection standard",
            "contexts": ["1926.501 covers fall protection"],
            "ground_truth": "fall protection",
            "trust_score": 72,
            "category": "keyword",
            "ground_truth_chunk_ids": ["c_100"],
            "retrieved_chunk_ids": ["c_100", "c_101"],
        },
        {
            "question": "Harness anchoring at height",
            "answer": "Use certified anchor points",
            "contexts": ["Anchor points must be rated for 5000 lbs"],
            "ground_truth": "anchor",
            "trust_score": 40,
            "category": "hybrid",
            "ground_truth_chunk_ids": ["c_200"],
            "retrieved_chunk_ids": ["c_201", "c_202"],  # Miss
        },
    ]


def test_hit5_by_category_computation(sample_rows):
    """Verify hit@5 computation per category."""
    hit = compute_hit_at_5(sample_rows)

    assert hit["semantic"] == 1.0  # c_042 found
    assert hit["keyword"] == 1.0  # c_100 found
    assert hit["hybrid"] == 0.0  # c_200 not found
    assert hit["overall"] == pytest.approx(2 / 3)


def test_hit5_empty_category():
    """Empty input gives overall 0."""
    assert compute_hit_at_5([]) == {"overall": 0.0}


@pytest.mark.asyncio
async def test_ragas_pipeline_mock_endpoint(tmp_path):
    """Mock httpx calls and verify row structure."""
    # Create a minimal dataset
    dataset = {
        "version": "1.0",
        "queries": [
            {
                "id": "Q001",
                "text": "Test query",
                "category": "semantic",
                "ground_truth_chunk_ids": ["c_001"],
                "expected_answer_substring": "test answer",
            }
        ],
    }
    dataset_path = tmp_path / "test_queries.json"
    dataset_path.write_text(json.dumps(dataset))

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "answer": "The test answer is here",
        "sources": [
            {"content": "Source content", "chunk_id": "c_001", "page": 1}
        ],
        "trust_score": 78,
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("trustrag_eval.ragas_pipeline.httpx.AsyncClient", return_value=mock_client):
        rows = await collect_results(
            endpoint="http://mock:8000",
            dataset_path=str(dataset_path),
        )

    assert len(rows) == 1
    row = rows[0]
    assert row["question"] == "Test query"
    assert row["answer"] == "The test answer is here"
    assert row["contexts"] == ["Source content"]
    assert row["trust_score"] == 78
    assert row["category"] == "semantic"
    assert row["ground_truth_chunk_ids"] == ["c_001"]
    assert row["retrieved_chunk_ids"] == ["c_001"]


def test_aggregates_all_three_metric_types(sample_rows):
    """Verify that compute_hit_at_5 + trust_metrics produce expected keys."""
    from trustrag_eval.trust_metrics import compute_trust_metrics

    hit = compute_hit_at_5(sample_rows)
    trust = compute_trust_metrics(sample_rows)

    # Hit@5 keys
    assert "overall" in hit
    assert "semantic" in hit

    # Trust keys
    assert "p25" in trust
    assert "p50" in trust
    assert "p75" in trust
    assert "mean" in trust
    assert "flagged_pct" in trust
