"""Tests for trust_verifier — embedding reuse optimization."""

import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock fastembed before importing trust_verifier (fastembed not installed locally)
sys.modules.setdefault("fastembed", MagicMock())


@pytest.mark.asyncio
async def test_source_agreement_uses_db_embeddings_not_recompute():
    """When sources come with 'embedding' field (from hybrid_search),
    _compute_source_agreement MUST NOT call embed_batch."""
    from services.trust_verifier import _compute_source_agreement

    # Sources already have embeddings (as hybrid_search now returns)
    sources = [
        {"content": "chunk 1", "embedding": [0.1] * 384},
        {"content": "chunk 2", "embedding": [0.2] * 384},
        {"content": "chunk 3", "embedding": [0.15] * 384},
    ]

    with patch("services.trust_verifier.embed_batch", new=AsyncMock()) as mock_embed:
        result = await _compute_source_agreement(sources)

    assert 0.0 <= result <= 1.0
    mock_embed.assert_not_called(), "Should reuse DB embeddings, not re-embed"


@pytest.mark.asyncio
async def test_source_agreement_falls_back_to_embed_batch_when_no_embeddings():
    """When sources lack 'embedding' field, _compute_source_agreement falls back to embed_batch."""
    from services.trust_verifier import _compute_source_agreement

    sources = [
        {"content": "chunk 1"},
        {"content": "chunk 2"},
    ]

    mock_embeddings = [[0.1] * 384, [0.2] * 384]
    with patch("services.trust_verifier.embed_batch", new=AsyncMock(return_value=mock_embeddings)) as mock_embed:
        result = await _compute_source_agreement(sources)

    assert 0.0 <= result <= 1.0
    mock_embed.assert_called_once()
