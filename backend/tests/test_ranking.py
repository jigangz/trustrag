"""Unit tests for RRF ranking (P2-2)."""

import pytest

from services.ranking import rrf_fuse


class TestRRFEmpty:
    def test_rrf_empty_rankings(self):
        """rrf_fuse([]) returns empty list."""
        result = rrf_fuse([])
        assert result == []

    def test_rrf_empty_inner_rankings(self):
        """rrf_fuse([[], []]) returns empty list."""
        result = rrf_fuse([[], []])
        assert result == []


class TestRRFSingleRanking:
    def test_rrf_single_ranking(self):
        """One ranking produces scores in descending order by 1/(k+rank+1)."""
        result = rrf_fuse([["a", "b", "c"]])
        ids = [doc_id for doc_id, _ in result]
        scores = [score for _, score in result]

        assert ids == ["a", "b", "c"]
        # Verify scores follow 1/(60+rank+1)
        assert scores[0] == pytest.approx(1 / 61)
        assert scores[1] == pytest.approx(1 / 62)
        assert scores[2] == pytest.approx(1 / 63)


class TestRRFTwoRankingsSameOrder:
    def test_rrf_two_rankings_same_order(self):
        """Same order in both rankings doubles scores, preserves order."""
        result = rrf_fuse([["a", "b"], ["a", "b"]])
        ids = [doc_id for doc_id, _ in result]
        scores = [score for _, score in result]

        assert ids == ["a", "b"]
        # Scores are doubled
        assert scores[0] == pytest.approx(2 / 61)
        assert scores[1] == pytest.approx(2 / 62)


class TestRRFDisjointRankings:
    def test_rrf_disjoint_rankings(self):
        """Disjoint rankings: all docs included, same-position docs get same score."""
        result = rrf_fuse([["a", "b"], ["c", "d"]])
        ids = [doc_id for doc_id, _ in result]
        scores = [score for _, score in result]

        # All 4 docs present
        assert set(ids) == {"a", "b", "c", "d"}
        # First-ranked docs from each list share same score
        assert scores[0] == pytest.approx(1 / 61)
        assert scores[1] == pytest.approx(1 / 61)
        # Second-ranked docs share same score
        assert scores[2] == pytest.approx(1 / 62)
        assert scores[3] == pytest.approx(1 / 62)


class TestRRFOverlapBoosts:
    def test_rrf_overlap_boosts(self):
        """Doc appearing in both rankings scores higher than disjoint docs."""
        # "a" appears in both, "b" and "c" only in one each
        result = rrf_fuse([["a", "b"], ["a", "c"]])
        scores_dict = dict(result)

        # "a" gets boosted (appears in both)
        assert scores_dict["a"] > scores_dict["b"]
        assert scores_dict["a"] > scores_dict["c"]
        # "a" score = 1/61 + 1/61 = 2/61
        assert scores_dict["a"] == pytest.approx(2 / 61)
        # "b" and "c" each get 1/62
        assert scores_dict["b"] == pytest.approx(1 / 62)
        assert scores_dict["c"] == pytest.approx(1 / 62)


class TestRRFKParameter:
    def test_rrf_k_parameter_tunes_smoothing(self):
        """Different k values produce different scores but consistent ordering."""
        rankings = [["a", "b", "c"], ["b", "a", "c"]]

        result_k1 = rrf_fuse(rankings, k=1)
        result_k100 = rrf_fuse(rankings, k=100)

        ids_k1 = [doc_id for doc_id, _ in result_k1]
        ids_k100 = [doc_id for doc_id, _ in result_k100]

        # Ordering should be consistent (a and b both appear in both lists
        # but at different positions; c is always last)
        assert ids_k1[-1] == "c"
        assert ids_k100[-1] == "c"

        # Scores with k=1 are larger magnitude than k=100
        scores_k1 = dict(result_k1)
        scores_k100 = dict(result_k100)
        assert scores_k1["a"] > scores_k100["a"]
        assert scores_k1["b"] > scores_k100["b"]


class TestRRFPurity:
    def test_does_not_mutate_inputs(self):
        """rrf_fuse does not mutate the input rankings."""
        ranking1 = ["a", "b", "c"]
        ranking2 = ["d", "e"]
        original1 = ranking1.copy()
        original2 = ranking2.copy()

        rrf_fuse([ranking1, ranking2])

        assert ranking1 == original1
        assert ranking2 == original2
