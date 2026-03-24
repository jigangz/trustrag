"""Database and API models for TrustRAG."""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel


# --- SQLAlchemy ORM Models ---

class Base(DeclarativeBase):
    pass


class Document(Base):
    """Uploaded document metadata."""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """Document chunk with embedding vector."""
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(Vector(384))  # BAAI/bge-small-en-v1.5 dimension

    document = relationship("Document", back_populates="chunks")


class AuditLog(Base):
    """Full audit trail for every query."""
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=False)
    confidence_level = Column(String(10), nullable=False)  # high, medium, low
    sources = Column(JSONB)  # [{doc_id, filename, page, text, similarity}]
    hallucination_flags = Column(JSONB)  # [{sentence, reason}]
    consistency_check = Column(JSONB)  # {consistent, score, variants}
    score_breakdown = Column(JSONB)  # {retrieval, source_count, agreement, hallucination}
    created_at = Column(DateTime, default=datetime.utcnow)


# --- Pydantic API Schemas ---

class QueryRequest(BaseModel):
    """Incoming question from a user."""
    question: str
    enable_consistency_check: bool = False


class SourceResponse(BaseModel):
    """A single source citation."""
    document: str
    page: int
    text: str
    similarity: float


class ConfidenceResponse(BaseModel):
    """Trust score with full breakdown."""
    score: float
    level: str  # "high", "medium", "low"
    breakdown: dict


class QueryResponse(BaseModel):
    """Complete answer with trust verification."""
    answer: str
    confidence: ConfidenceResponse
    sources: list[SourceResponse]
    hallucination_check: dict
    consistency_check: dict | None
    audit_id: str


class DocumentResponse(BaseModel):
    """Document metadata for API responses."""
    id: str
    filename: str
    uploaded_at: datetime
    total_pages: int
    total_chunks: int
