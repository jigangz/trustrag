"""pgvector + tsvector operations for hybrid document retrieval."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from config import settings
from services.ranking import rrf_fuse


async def store_chunks(session: AsyncSession, document_id: str, chunks: list[dict]):
    """
    Store document chunks with their embeddings in pgvector.

    Args:
        session: Database session
        document_id: Parent document UUID
        chunks: List of {content, page_number, chunk_index, embedding}
    """
    for chunk in chunks:
        await session.execute(
            text("""
                INSERT INTO chunks (id, document_id, content, page_number, chunk_index, embedding)
                VALUES (gen_random_uuid(), :doc_id, :content, :page, :idx, :embedding)
            """),
            {
                "doc_id": document_id,
                "content": chunk["content"],
                "page": chunk["page_number"],
                "idx": chunk["chunk_index"],
                "embedding": str(chunk["embedding"]),
            },
        )
    await session.commit()


async def search_similar(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """
    Find the most similar chunks to a query embedding using cosine distance.

    Returns:
        List of {chunk_id, document_id, filename, content, page_number, similarity}
        sorted by similarity descending.
    """
    result = await session.execute(
        text("""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                d.filename,
                c.content,
                c.page_number,
                1 - (c.embedding <=> :embedding) AS similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            ORDER BY c.embedding <=> :embedding
            LIMIT :top_k
        """),
        {"embedding": str(query_embedding), "top_k": top_k},
    )
    rows = result.mappings().all()
    return [
        {
            "chunk_id": str(row["chunk_id"]),
            "document_id": str(row["document_id"]),
            "filename": row["filename"],
            "content": row["content"],
            "page_number": row["page_number"],
            "similarity": float(row["similarity"]),
        }
        for row in rows
    ]


async def hybrid_search(
    session: AsyncSession,
    query_embedding: list[float],
    query_text: str,
    top_k: int | None = None,
) -> list[dict]:
    """
    Hybrid retrieval combining semantic (pgvector) and keyword (tsvector) search
    via Reciprocal Rank Fusion.

    When HYBRID_ENABLED=false, falls back to pure semantic search (v0.1 behavior).

    Args:
        session: Database session
        query_embedding: Pre-computed query embedding vector
        query_text: Raw query string for keyword search
        top_k: Number of final results (defaults to settings.final_top_k)

    Returns:
        List of chunk dicts sorted by fused relevance.
    """
    top_k = top_k or settings.final_top_k

    if not settings.hybrid_enabled:
        return await search_similar(session, query_embedding, top_k=top_k)

    # Run semantic and keyword search in parallel
    semantic_results, keyword_results = await asyncio.gather(
        search_similar(session, query_embedding, top_k=settings.semantic_candidates),
        _keyword_search(session, query_text, limit=settings.keyword_candidates),
    )

    # Fuse rankings via RRF
    semantic_ids = [c["chunk_id"] for c in semantic_results]
    keyword_ids = [c["chunk_id"] for c in keyword_results]

    fused = rrf_fuse([semantic_ids, keyword_ids], k=settings.rrf_k)
    top_ids = [doc_id for doc_id, _ in fused[:top_k]]

    if not top_ids:
        return []

    # Build lookup preferring semantic_results (has real cosine similarity).
    # If chunk only appeared in keyword path, fall back to its ts_rank.
    # Previous impl used _fetch_chunks_by_ids which zeroed out similarity —
    # that made UI and benchmarks under-report the retrieval quality.
    lookup = {c["chunk_id"]: c for c in keyword_results}
    for c in semantic_results:
        lookup[c["chunk_id"]] = c  # semantic wins (real cosine similarity)

    # Any missing ids (rare) → fetch via _fetch_chunks_by_ids
    missing = [cid for cid in top_ids if cid not in lookup]
    if missing:
        for chunk in await _fetch_chunks_by_ids(session, missing):
            lookup[chunk["chunk_id"]] = chunk

    return [lookup[cid] for cid in top_ids if cid in lookup]


async def _keyword_search(
    session: AsyncSession,
    query: str,
    limit: int,
) -> list[dict]:
    """
    Full-text keyword search using tsvector + GIN index.

    Returns:
        List of {chunk_id, document_id, filename, content, page_number, similarity}
        sorted by ts_rank descending.
    """
    result = await session.execute(
        text("""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                d.filename,
                c.content,
                c.page_number,
                ts_rank(c.content_tsv, plainto_tsquery('english', :query)) AS similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.content_tsv @@ plainto_tsquery('english', :query)
            ORDER BY similarity DESC
            LIMIT :limit
        """),
        {"query": query, "limit": limit},
    )
    rows = result.mappings().all()
    return [
        {
            "chunk_id": str(row["chunk_id"]),
            "document_id": str(row["document_id"]),
            "filename": row["filename"],
            "content": row["content"],
            "page_number": row["page_number"],
            "similarity": float(row["similarity"]),
        }
        for row in rows
    ]


async def _fetch_chunks_by_ids(
    session: AsyncSession,
    chunk_ids: list[str],
) -> list[dict]:
    """
    Fetch full chunk records by ID, preserving the order of chunk_ids.

    Args:
        session: Database session
        chunk_ids: Ordered list of chunk UUIDs

    Returns:
        List of chunk dicts in the same order as chunk_ids.
    """
    if not chunk_ids:
        return []

    result = await session.execute(
        text("""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                d.filename,
                c.content,
                c.page_number,
                0.0 AS similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.id = ANY(CAST(:ids AS uuid[]))
        """),
        {"ids": chunk_ids},
    )
    rows = result.mappings().all()

    # Build lookup and preserve fused order
    by_id = {
        str(row["chunk_id"]): {
            "chunk_id": str(row["chunk_id"]),
            "document_id": str(row["document_id"]),
            "filename": row["filename"],
            "content": row["content"],
            "page_number": row["page_number"],
            "similarity": float(row["similarity"]),
        }
        for row in rows
    }
    return [by_id[cid] for cid in chunk_ids if cid in by_id]
