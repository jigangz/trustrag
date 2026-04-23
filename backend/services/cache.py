"""Postgres-backed query cache for hot-query sub-200ms response.

Behind `query_cache_enabled` flag. Key = sha256(normalized_question|top_k).
TTL = 24h by default. Invalidated on document upload.
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings


def _hash_query(question: str, top_k: int) -> str:
    """Produce deterministic cache key from question + top_k.

    Normalization: lowercase, strip, collapse internal whitespace.
    """
    normalized = question.lower().strip()
    normalized = " ".join(normalized.split())
    return hashlib.sha256(f"{normalized}|{top_k}".encode("utf-8")).hexdigest()


async def get(session: AsyncSession, question: str, top_k: int) -> dict | None:
    """Fetch cached response if exists and not expired. Increments hit counter."""
    if not settings.query_cache_enabled:
        return None

    h = _hash_query(question, top_k)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.query_cache_ttl_hours)

    result = await session.execute(
        text("""
            SELECT response_json
            FROM query_cache
            WHERE question_hash = :h AND created_at > :cutoff
        """),
        {"h": h, "cutoff": cutoff},
    )
    row = result.first()
    if row is None:
        return None

    # Atomic hit counter update
    await session.execute(
        text("""
            UPDATE query_cache
            SET hit_count = hit_count + 1, last_hit_at = NOW()
            WHERE question_hash = :h
        """),
        {"h": h},
    )
    await session.commit()
    return row[0]


async def set(session: AsyncSession, question: str, top_k: int, response: dict) -> None:
    """Store response in cache. Upserts on conflict."""
    if not settings.query_cache_enabled:
        return

    h = _hash_query(question, top_k)
    await session.execute(
        text("""
            INSERT INTO query_cache (question_hash, response_json)
            VALUES (:h, CAST(:r AS jsonb))
            ON CONFLICT (question_hash) DO UPDATE SET
                response_json = EXCLUDED.response_json,
                created_at = NOW(),
                hit_count = 0
        """),
        {"h": h, "r": json.dumps(response)},
    )
    await session.commit()


async def clear_all(session: AsyncSession) -> int:
    """Clear entire cache. Called on document upload. Returns rows deleted."""
    result = await session.execute(text("DELETE FROM query_cache"))
    await session.commit()
    return result.rowcount or 0
