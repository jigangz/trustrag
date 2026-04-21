"""Tests for the benchmark runner (P2-4).

Validates output shape and hit@5 computation logic without
requiring a running server.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add eval dir to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from run_benchmark import load_queries, run_benchmark, print_comparison_table


class TestLoadQueries:
    """Tests for query loading and validation."""

    def test_loads_30_queries(self):
        queries = load_queries()
        assert len(queries) == 30

    def test_each_query_has_required_fields(self):
        queries = load_queries()
        required_fields = {"id", "text", "category", "ground_truth_chunk_ids", "expected_answer_substring"}
        for q in queries:
            missing = required_fields - set(q.keys())
            assert not missing, f"Query {q.get('id', '?')} missing fields: {missing}"

    def test_category_distribution(self):
        queries = load_queries()
        categories = {}
        for q in queries:
            categories[q["category"]] = categories.get(q["category"], 0) + 1
        assert categories == {"semantic": 10, "keyword": 10, "hybrid": 10}

    def test_query_ids_unique(self):
        queries = load_queries()
        ids = [q["id"] for q in queries]
        assert len(ids) == len(set(ids)), "Duplicate query IDs found"

    def test_ground_truth_chunk_ids_non_empty(self):
        queries = load_queries()
        for q in queries:
            assert len(q["ground_truth_chunk_ids"]) > 0, f"Query {q['id']} has empty ground_truth"


class TestBenchmarkRunner:
    """Tests for benchmark runner output shape and hit@5 logic."""

    @pytest.mark.asyncio
    async def test_benchmark_produces_expected_shape(self):
        """Mock the API and verify output structure."""
        mock_response = {
            "answer": "Fall protection is required above 6 feet.",
            "trust_score": 0.85,
            "sources": [
                {"chunk_id": "c_042", "content": "...", "similarity": 0.9},
                {"chunk_id": "c_100", "content": "...", "similarity": 0.8},
            ],
        }

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("run_benchmark.httpx.AsyncClient", return_value=mock_client):
            results = await run_benchmark(mode="hybrid")

        # Check top-level keys
        assert "mode" in results
        assert "date" in results
        assert "overall_hit_at_5" in results
        assert "hit_at_5_by_category" in results
        assert "total_queries" in results
        assert "queries" in results

        # Check category keys
        assert set(results["hit_at_5_by_category"].keys()) == {"semantic", "keyword", "hybrid"}

        # Check total
        assert results["total_queries"] == 30
        assert results["mode"] == "hybrid"

    @pytest.mark.asyncio
    async def test_hit5_computation_correct(self):
        """Verify hit@5 is True when ground_truth_chunk_id is in retrieved set."""
        # First query expects c_042 - our mock returns it
        mock_response_hit = {
            "answer": "...",
            "trust_score": 0.8,
            "sources": [
                {"chunk_id": "c_042", "content": "...", "similarity": 0.9},
                {"chunk_id": "c_999", "content": "...", "similarity": 0.7},
            ],
        }
        # Other queries won't match
        mock_response_miss = {
            "answer": "...",
            "trust_score": 0.5,
            "sources": [
                {"chunk_id": "c_999", "content": "...", "similarity": 0.6},
            ],
        }

        call_count = {"n": 0}

        mock_client = AsyncMock()

        async def mock_post(*args, **kwargs):
            call_count["n"] += 1
            resp = MagicMock()
            # Return a hit for the first query only
            if call_count["n"] == 1:
                resp.json.return_value = mock_response_hit
            else:
                resp.json.return_value = mock_response_miss
            resp.raise_for_status = MagicMock()
            return resp

        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("run_benchmark.httpx.AsyncClient", return_value=mock_client):
            results = await run_benchmark(mode="hybrid")

        # First query should be a hit (c_042 is in ground truth for Q001)
        q001 = next(r for r in results["queries"] if r["query_id"] == "Q001")
        assert q001["hit@5"] is True

        # Most others should be misses
        hits = sum(1 for r in results["queries"] if r["hit@5"])
        assert hits >= 1  # At least the first one hit


class TestComparisonTable:
    """Test the comparison table formatting."""

    def test_print_comparison_table_returns_string(self):
        hybrid = {
            "overall_hit_at_5": 0.73,
            "hit_at_5_by_category": {"semantic": 0.8, "keyword": 0.7, "hybrid": 0.7},
        }
        semantic = {
            "overall_hit_at_5": 0.60,
            "hit_at_5_by_category": {"semantic": 0.8, "keyword": 0.4, "hybrid": 0.6},
        }
        table = print_comparison_table(hybrid, semantic)
        assert "OVERALL" in table
        assert "semantic" in table
        assert "keyword" in table
        assert "hybrid" in table
