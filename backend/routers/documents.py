"""Document upload and management endpoints."""

import uuid

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from database import get_session
from config import settings
from models import DocumentResponse
from services.document_processor import parse_pdf, chunk_text
from services.embedding import embed_batch
from services.vector_store import store_chunks

router = APIRouter()

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """Upload a PDF document, parse it into chunks, and generate embeddings."""
    # 1. Validate file is PDF
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # 2. Parse with pdfplumber
    try:
        pages = await parse_pdf(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {e}")

    if not pages:
        raise HTTPException(status_code=400, detail="No text content found in PDF")

    # 3. Chunk text
    chunks = chunk_text(pages, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks generated from PDF")

    # 4. Generate embeddings via fastembed (local)
    try:
        texts = [c["content"] for c in chunks]
        embeddings = await embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding API error: {e}")

    # 5. Store document + chunks in DB
    doc_id = str(uuid.uuid4())
    await session.execute(
        text("""
            INSERT INTO documents (id, filename, total_pages, total_chunks)
            VALUES (:id, :filename, :pages, :chunks)
        """),
        {
            "id": doc_id,
            "filename": file.filename,
            "pages": len(pages),
            "chunks": len(chunks),
        },
    )
    await session.commit()

    await store_chunks(session, doc_id, chunks)

    return {
        "id": doc_id,
        "filename": file.filename,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "message": f"Successfully processed {file.filename}",
    }


@router.get("/", response_model=list[DocumentResponse])
async def list_documents(session: AsyncSession = Depends(get_session)):
    """List all uploaded documents with metadata."""
    result = await session.execute(
        text("SELECT id, filename, uploaded_at, total_pages, total_chunks FROM documents ORDER BY uploaded_at DESC")
    )
    rows = result.mappings().all()
    return [
        DocumentResponse(
            id=str(row["id"]),
            filename=row["filename"],
            uploaded_at=row["uploaded_at"],
            total_pages=row["total_pages"],
            total_chunks=row["total_chunks"],
        )
        for row in rows
    ]


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete a document and all its chunks."""
    # Verify document exists
    result = await session.execute(
        text("SELECT id FROM documents WHERE id = :id"),
        {"id": document_id},
    )
    if not result.first():
        raise HTTPException(status_code=404, detail="Document not found")

    # Cascade delete: chunks first, then document
    await session.execute(text("DELETE FROM chunks WHERE document_id = :id"), {"id": document_id})
    await session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})
    await session.commit()

    return {"message": "Document deleted", "id": document_id}
