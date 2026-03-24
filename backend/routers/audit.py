"""Audit trail endpoints for query history."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from database import get_session

router = APIRouter()


@router.get("/")
async def list_audit_logs(
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    """List recent query audit logs, most recent first."""
    result = await session.execute(
        text("""
            SELECT id, query, answer, confidence_score, confidence_level, created_at
            FROM audit_logs
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        {"limit": limit, "offset": offset},
    )
    rows = result.mappings().all()
    return [
        {
            "id": str(row["id"]),
            "query": row["query"],
            "answer": row["answer"][:200] + "..." if len(row["answer"]) > 200 else row["answer"],
            "confidence_score": row["confidence_score"],
            "confidence_level": row["confidence_level"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in rows
    ]


@router.get("/{audit_id}")
async def get_audit_detail(
    audit_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get full audit details for a specific query, including sources and verification results."""
    result = await session.execute(
        text("SELECT * FROM audit_logs WHERE id = :id"),
        {"id": audit_id},
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Audit log not found")

    return {
        "id": str(row["id"]),
        "query": row["query"],
        "answer": row["answer"],
        "confidence_score": row["confidence_score"],
        "confidence_level": row["confidence_level"],
        "sources": row["sources"],
        "hallucination_flags": row["hallucination_flags"],
        "consistency_check": row["consistency_check"],
        "score_breakdown": row["score_breakdown"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }
