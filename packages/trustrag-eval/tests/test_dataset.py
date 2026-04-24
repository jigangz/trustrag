"""Tests for dataset loading."""

import json
from pathlib import Path

from trustrag_eval.dataset import load_synthetic_queries


def _create_test_dataset(tmp_path: Path) -> Path:
    """Create a minimal test dataset file."""
    data = {
        "version": "1.0",
        "queries": [
            {
                "id": f"Q{i:03d}",
                "text": f"Test query {i}",
                "category": ["semantic", "keyword", "hybrid"][i % 3],
                "ground_truth_chunk_ids": [f"c_{i:03d}"],
                "expected_answer_substring": f"answer {i}",
            }
            for i in range(30)
        ],
    }
    filepath = tmp_path / "synthetic_queries.json"
    filepath.write_text(json.dumps(data), encoding="utf-8")
    return filepath


def test_load_synthetic_queries_parses_30(tmp_path):
    """Load dataset and verify 30 queries with all 3 categories present."""
    filepath = _create_test_dataset(tmp_path)
    queries = load_synthetic_queries(filepath)

    assert len(queries) == 30
    categories = {q.category for q in queries}
    assert categories == {"semantic", "keyword", "hybrid"}


def test_load_synthetic_queries_fields(tmp_path):
    """Verify all fields are correctly parsed."""
    filepath = _create_test_dataset(tmp_path)
    queries = load_synthetic_queries(filepath)

    q = queries[0]
    assert q.id == "Q000"
    assert q.text == "Test query 0"
    assert q.category == "semantic"
    assert q.ground_truth_chunk_ids == ["c_000"]
    assert q.expected_answer_substring == "answer 0"


def test_load_real_dataset():
    """Load the actual synthetic_queries.json from the repo."""
    repo_path = Path(__file__).parent.parent.parent.parent.parent / "eval" / "synthetic_queries.json"
    if not repo_path.exists():
        return  # Skip if not running from repo root

    queries = load_synthetic_queries(repo_path)
    assert len(queries) == 30
    categories = {q.category for q in queries}
    assert "semantic" in categories
    assert "keyword" in categories
    assert "hybrid" in categories
