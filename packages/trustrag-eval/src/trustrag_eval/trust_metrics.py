"""Trust-specific metrics for TrustRAG evaluation."""

import statistics
from typing import TypedDict


class TrustDistribution(TypedDict):
    p25: float
    p50: float
    p75: float
    mean: float
    flagged_pct: float  # % with trust_score < 50


def compute_trust_metrics(results: list[dict]) -> TrustDistribution:
    """Compute trust score distribution from query results.

    Args:
        results: List of dicts, each with a "trust_score" key (0-100).

    Returns:
        TrustDistribution with percentiles, mean, and flagged percentage.

    Raises:
        ValueError: If no valid trust scores found.
    """
    scores = [r["trust_score"] for r in results if r.get("trust_score") is not None]
    if not scores:
        raise ValueError("No valid trust scores found in results")

    return TrustDistribution(
        p25=statistics.quantiles(scores, n=4)[0],
        p50=statistics.median(scores),
        p75=statistics.quantiles(scores, n=4)[2],
        mean=statistics.mean(scores),
        flagged_pct=sum(1 for s in scores if s < 50) / len(scores),
    )
