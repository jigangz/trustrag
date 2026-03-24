"""pgvector operations for storing and retrieving document chunks."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


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
