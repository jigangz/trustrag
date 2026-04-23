"""Tests for Postgres-backed query cache service."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from services import cache


def test_hash_query_normalizes_case_and_whitespace():
    h1 = cache._hash_query("What is fall protection?", top_k=5)
    h2 = cache._hash_query("what is fall protection?", top_k=5)
    h3 = cache._hash_query("  What  is   fall protection? ", top_k=5)
    assert h1 == h2 == h3


def test_hash_query_different_top_k_differs():
    h1 = cache._hash_query("question", top_k=5)
    h2 = cache._hash_query("question", top_k=10)
    assert h1 != h2


def test_hash_query_different_text_differs():
    h1 = cache._hash_query("question A", top_k=5)
    h2 = cache._hash_query("question B", top_k=5)
    assert h1 != h2


@pytest.mark.asyncio
async def test_cache_get_returns_none_when_flag_disabled(monkeypatch):
    """When QUERY_CACHE_ENABLED=false, get() returns None without DB query."""
    from config import settings
    monkeypatch.setattr(settings, "query_cache_enabled", False)

    mock_session = AsyncMock()
    result = await cache.get(mock_session, "test q", top_k=5)
    assert result is None
    mock_session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_cache_set_then_get_round_trip(monkeypatch):
    """Set a response, then get returns it. Verifies execute calls are correct."""
    from config import settings
    monkeypatch.setattr(settings, "query_cache_enabled", True)

    response_data = {"answer": "test answer", "sources": [], "trust_score": 85}

    # --- Test set ---
    set_session = AsyncMock()
    await cache.set(set_session, "fall protection query", top_k=5, response=response_data)
    # Should have called execute (INSERT) + commit
    assert set_session.execute.call_count == 1
    set_session.commit.assert_called_once()

    # Verify the hash used in the INSERT
    h = cache._hash_query("fall protection query", 5)
    set_call_args = set_session.execute.call_args
    params = set_call_args[1] if set_call_args[1] else set_call_args[0][1]
    assert params["h"] == h
    assert json.loads(params["r"]) == response_data

    # --- Test get (cache hit) ---
    get_session = AsyncMock()
    # Simulate DB returning the cached row
    mock_row = MagicMock()
    mock_row.__getitem__ = lambda self, idx: response_data
    mock_result = MagicMock()
    mock_result.first.return_value = mock_row
    get_session.execute.return_value = mock_result

    result = await cache.get(get_session, "fall protection query", top_k=5)
    assert result == response_data
    # Should have called execute twice: SELECT + UPDATE hit_count
    assert get_session.execute.call_count == 2
    get_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_cache_clear_all_removes_entries(monkeypatch):
    """clear_all issues DELETE and returns rowcount."""
    from config import settings
    monkeypatch.setattr(settings, "query_cache_enabled", True)

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.rowcount = 3
    mock_session.execute.return_value = mock_result

    count = await cache.clear_all(mock_session)
    assert count == 3
    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_cache_set_noop_when_flag_disabled(monkeypatch):
    """When QUERY_CACHE_ENABLED=false, set() does nothing."""
    from config import settings
    monkeypatch.setattr(settings, "query_cache_enabled", False)

    mock_session = AsyncMock()
    await cache.set(mock_session, "test q", top_k=5, response={"a": 1})
    mock_session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_cache_get_returns_none_on_miss(monkeypatch):
    """When no cached row exists, get() returns None without updating hit_count."""
    from config import settings
    monkeypatch.setattr(settings, "query_cache_enabled", True)

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.first.return_value = None
    mock_session.execute.return_value = mock_result

    result = await cache.get(mock_session, "unknown question", top_k=5)
    assert result is None
    # Only SELECT, no UPDATE
    assert mock_session.execute.call_count == 1
    mock_session.commit.assert_not_called()
