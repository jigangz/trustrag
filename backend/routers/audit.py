"""Audit trail endpoints for query history."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session

router = APIRouter()


@router.get("/")
async def list_audit_logs(
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    """List recent query audit logs, most recent first."""
    # TODO: Query audit_logs table, return paginated list
    pass


@router.get("/{audit_id}")
async def get_audit_detail(
    audit_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get full audit details for a specific query, including sources and verification results."""
    # TODO: Fetch single audit log with all metadata
    pass
