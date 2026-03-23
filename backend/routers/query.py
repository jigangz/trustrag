"""Question answering with trust verification."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from models import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Process a question through the full TrustRAG pipeline:

    1. Embed the question
    2. Retrieve top-k relevant chunks from pgvector
    3. Generate answer with source attribution via LLM
    4. Run trust verification:
       - Confidence scoring (retrieval quality + source agreement)
       - Hallucination detection (secondary LLM check)
       - Optional: answer consistency check (3 rephrasings)
    5. Log everything to audit trail
    6. Return answer with full trust metadata
    """
    # TODO: Implement full RAG + trust pipeline
    # See services/rag_engine.py, trust_verifier.py, consistency_checker.py
    pass
