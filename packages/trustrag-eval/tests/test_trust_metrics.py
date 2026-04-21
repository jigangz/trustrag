"""Tests for trust metrics computation."""

import pytest

from trustrag_eval.trust_metrics import compute_trust_metrics


def test_compute_trust_metrics_returns_expected_shape():
    """Golden inputs produce expected distribution values."""
    results = [
        {"trust_score": 90},
        {"trust_score": 80},
        {"trust_score": 70},
        {"trust_score": 60},
        {"trust_score": 50},
    ]
    dist = compute_trust_metrics(results)

    assert "p25" in dist
    assert "p50" in dist
    assert "p75" in dist
    assert "mean" in dist
    assert "flagged_pct" in dist

    assert dist["p50"] == 70
    assert dist["mean"] == 70.0
    assert dist["flagged_pct"] == 0.0  # All >= 50


def test_compute_trust_metrics_flagged_pct():
    """50-50 split above/below 50 gives flagged_pct == 0.5."""
    results = [
        {"trust_score": 80},
        {"trust_score": 70},
        {"trust_score": 40},
        {"trust_score": 30},
    ]
    dist = compute_trust_metrics(results)

    assert dist["flagged_pct"] == 0.5


def test_compute_trust_metrics_all_flagged():
    """All scores below 50 gives flagged_pct == 1.0."""
    results = [{"trust_score": s} for s in [10, 20, 30, 40]]
    dist = compute_trust_metrics(results)
    assert dist["flagged_pct"] == 1.0


def test_compute_trust_metrics_skips_none():
    """Results with None trust_score are excluded."""
    results = [
        {"trust_score": 80},
        {"trust_score": None},
        {"trust_score": 60},
        {"trust_score": 70},
        {"other_field": "no trust_score key"},
    ]
    dist = compute_trust_metrics(results)
    assert dist["mean"] == 70.0


def test_compute_trust_metrics_empty_raises():
    """Empty results raise ValueError."""
    with pytest.raises(ValueError, match="No valid trust scores"):
        compute_trust_metrics([])
