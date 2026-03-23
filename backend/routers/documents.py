"""Document upload and management endpoints."""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session

router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """Upload a PDF document, parse it into chunks, and generate embeddings."""
    # TODO: Implement document processing pipeline
    # 1. Validate file is PDF
    # 2. Parse with pdfplumber
    # 3. Chunk text by paragraphs (~500 tokens each)
    # 4. Generate embeddings via OpenAI
    # 5. Store document + chunks + embeddings in pgvector
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/")
async def list_documents(session: AsyncSession = Depends(get_session)):
    """List all uploaded documents with metadata."""
    # TODO: Query documents table, return list
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete a document and all its chunks."""
    # TODO: Cascade delete document + chunks
    raise HTTPException(status_code=501, detail="Not implemented yet")
