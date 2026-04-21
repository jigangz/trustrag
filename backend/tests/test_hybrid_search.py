"""Integration tests for hybrid_search (P2-3).

Uses mocks for DB layer since these tests verify logic flow,
not actual database queries. DB integration is tested via the
migration test and manual verification.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest


# Sample chunk data for tests
CHUNK_A = {
    "chunk_id": "aaa-111",
    "document_id": "doc-1",
    "filename": "safety.pdf",
    "content": "Fall protection required above 6 feet",
    "page_number": 1,
    "similarity": 0.92,
}
CHUNK_B = {
    "chunk_id": "bbb-222",
    "document_id": "doc-1",
    "filename": "safety.pdf",
    "content": "OSHA 1926.501 scaffold requirements",
    "page_number": 2,
    "similarity": 0.85,
}
CHUNK_C = {
    "chunk_id": "ccc-333",
    "document_id": "doc-2",
    "filename": "regs.pdf",
    "content": "Personal protective equipment standards",
    "page_number": 5,
    "similarity": 0.78,
}


@pytest.fixture
def mock_session():
    """Create a mock AsyncSession."""
    session = AsyncMock()
    return session


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings to defaults before each test."""
    from config import settings
    original_hybrid = settings.hybrid_enabled
    original_semantic = settings.semantic_candidates
    original_keyword = settings.keyword_candidates
    original_rrf_k = settings.rrf_k
    original_top_k = settings.final_top_k
    yield
    settings.hybrid_enabled = original_hybrid
    settings.semantic_candidates = original_semantic
    settings.keyword_candidates = original_keyword
    settings.rrf_k = original_rrf_k
    settings.final_top_k = original_top_k


class TestHybridSearchBothPaths:
    @pytest.mark.asyncio
    async def test_hybrid_search_both_paths_return_results(self, mock_session):
        """When both semantic and keyword return results, fused output combines them."""
        from services.vector_store import hybrid_search

        with patch("services.vector_store.search_similar", new_callable=AsyncMock) as mock_sem, \
             patch("services.vector_store._keyword_search", new_callable=AsyncMock) as mock_kw, \
             patch("services.vector_store._fetch_chunks_by_ids", new_callable=AsyncMock) as mock_fetch:

            mock_sem.return_value = [CHUNK_A, CHUNK_B]
            mock_kw.return_value = [CHUNK_B, CHUNK_C]
            mock_fetch.return_value = [CHUNK_B, CHUNK_A, CHUNK_C]

            result = await hybrid_search(
                mock_session, [0.1] * 384, "fall protection", top_k=5
            )

            assert len(result) == 3
            # B appears in both rankings, should be boosted to top
            assert result[0]["chunk_id"] == "bbb-222"
            mock_sem.assert_called_once()
            mock_kw.assert_called_once()
            mock_fetch.assert_called_once()


class TestHybridSearchKeywordOnlyFallback:
    @pytest.mark.asyncio
    async def test_hybrid_search_keyword_only_fallback(self, mock_session):
        """When semantic returns nothing, keyword results still surface."""
        from services.vector_store import hybrid_search

        with patch("services.vector_store.search_similar", new_callable=AsyncMock) as mock_sem, \
             patch("services.vector_store._keyword_search", new_callable=AsyncMock) as mock_kw, \
             patch("services.vector_store._fetch_chunks_by_ids", new_callable=AsyncMock) as mock_fetch:

            mock_sem.return_value = []
            mock_kw.return_value = [CHUNK_C]
            mock_fetch.return_value = [CHUNK_C]

            result = await hybrid_search(
                mock_session, [0.1] * 384, "PPE standards", top_k=5
            )

            assert len(result) == 1
            assert result[0]["chunk_id"] == "ccc-333"


class TestHybridSearchSemanticOnlyFallback:
    @pytest.mark.asyncio
    async def test_hybrid_search_semantic_only_fallback(self, mock_session):
        """When keyword returns nothing, semantic results still surface."""
        from services.vector_store import hybrid_search

        with patch("services.vector_store.search_similar", new_callable=AsyncMock) as mock_sem, \
             patch("services.vector_store._keyword_search", new_callable=AsyncMock) as mock_kw, \
             patch("services.vector_store._fetch_chunks_by_ids", new_callable=AsyncMock) as mock_fetch:

            mock_sem.return_value = [CHUNK_A]
            mock_kw.return_value = []
            mock_fetch.return_value = [CHUNK_A]

            result = await hybrid_search(
                mock_session, [0.1] * 384, "the it is a", top_k=5
            )

            assert len(result) == 1
            assert result[0]["chunk_id"] == "aaa-111"


class TestHybridSearchBothEmpty:
    @pytest.mark.asyncio
    async def test_hybrid_search_both_empty_returns_empty_list(self, mock_session):
        """When both paths return nothing, result is empty list."""
        from services.vector_store import hybrid_search

        with patch("services.vector_store.search_similar", new_callable=AsyncMock) as mock_sem, \
             patch("services.vector_store._keyword_search", new_callable=AsyncMock) as mock_kw:

            mock_sem.return_value = []
            mock_kw.return_value = []

            result = await hybrid_search(
                mock_session, [0.1] * 384, "xyzzy", top_k=5
            )

            assert result == []


class TestHybridEnabledFalse:
    @pytest.mark.asyncio
    async def test_hybrid_enabled_false_preserves_v01_behavior(self, mock_session):
        """With hybrid_enabled=False, only semantic search is called (v0.1 path)."""
        from config import settings
        from services.vector_store import hybrid_search

        settings.hybrid_enabled = False

        with patch("services.vector_store.search_similar", new_callable=AsyncMock) as mock_sem, \
             patch("services.vector_store._keyword_search", new_callable=AsyncMock) as mock_kw:

            mock_sem.return_value = [CHUNK_A, CHUNK_B]

            result = await hybrid_search(
                mock_session, [0.1] * 384, "fall protection", top_k=5
            )

            # Only semantic called, keyword NOT called
            mock_sem.assert_called_once_with(mock_session, [0.1] * 384, top_k=5)
            mock_kw.assert_not_called()
            # Returns semantic results directly
            assert result == [CHUNK_A, CHUNK_B]


class TestHybridSearchUsesAsyncioGather:
    @pytest.mark.asyncio
    async def test_hybrid_search_uses_asyncio_gather(self, mock_session):
        """Semantic and keyword run in parallel — total time ≈ max, not sum."""
        from services.vector_store import hybrid_search

        async def slow_semantic(*args, **kwargs):
            await asyncio.sleep(0.1)
            return [CHUNK_A]

        async def slow_keyword(*args, **kwargs):
            await asyncio.sleep(0.1)
            return [CHUNK_B]

        with patch("services.vector_store.search_similar", side_effect=slow_semantic), \
             patch("services.vector_store._keyword_search", side_effect=slow_keyword), \
             patch("services.vector_store._fetch_chunks_by_ids", new_callable=AsyncMock) as mock_fetch:

            mock_fetch.return_value = [CHUNK_A, CHUNK_B]

            start = time.monotonic()
            result = await hybrid_search(
                mock_session, [0.1] * 384, "test query", top_k=5
            )
            elapsed = time.monotonic() - start

            # If run in parallel: ~0.1s. If sequential: ~0.2s.
            # Allow generous margin but must be well under sequential time.
            assert elapsed < 0.18, f"Took {elapsed:.3f}s — likely sequential, not parallel"
            assert len(result) == 2
