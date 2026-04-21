"""Reciprocal Rank Fusion (RRF) for combining multiple ranked result lists."""

from typing import Sequence


def rrf_fuse(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (Cormack et al. 2009).

    Combines multiple ranked lists into a single fused ranking by summing
    reciprocal ranks with a smoothing constant.

    Args:
        rankings: List of ranked doc_id lists (each ordered best to worst).
        k: Smoothing constant (default 60, paper empirical optimum).

    Returns:
        List of (doc_id, score) sorted by score descending.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
