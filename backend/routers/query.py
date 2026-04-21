"""Question answering with trust verification."""

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from database import get_session, async_session
from models import QueryRequest, QueryResponse, SourceResponse, ConfidenceResponse
from services.embedding import embed_text
from services.vector_store import search_similar, hybrid_search
from services.rag_engine import generate_answer
from services.trust_verifier import compute_trust_score
from services.consistency_checker import check_consistency

router = APIRouter()


async def _rag_answer_only(question: str) -> str:
    """Lightweight RAG pass for consistency checking — returns answer text only."""
    async with async_session() as session:
        q_emb = await embed_text(question)
        chunks = await search_similar(session, q_emb, top_k=5)
        if not chunks:
            return ""
        result = await generate_answer(question, chunks)
        return result["answer"]


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
    4. Run trust verification
    5. Log everything to audit trail
    6. Return answer with full trust metadata
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # 1. Embed question
    try:
        query_embedding = await embed_text(request.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding API error: {e}")

    # 2. Retrieve top-k chunks via hybrid search (pgvector + tsvector + RRF)
    # Falls back to pure semantic if HYBRID_ENABLED=false (see vector_store.hybrid_search)
    sources = await hybrid_search(
        session, query_embedding, request.question, top_k=5
    )
    if not sources:
        raise HTTPException(
            status_code=404,
            detail="No documents found. Please upload documents first.",
        )

    # 3. Generate answer
    try:
        rag_result = await generate_answer(request.question, sources)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM generation error: {e}")

    answer = rag_result["answer"]

    # 4. Trust verification
    trust_score = await compute_trust_score(answer, sources, query_embedding)

    # 5. Optional consistency check
    consistency_result = None
    if request.enable_consistency_check:
        consistency_result = await check_consistency(
            request.question, answer, _rag_answer_only
        )

    # 6. Audit log
    audit_id = str(uuid.uuid4())
    source_records = [
        {
            "document_id": s["document_id"],
            "filename": s["filename"],
            "page": s["page_number"],
            "text": s["content"][:500],
            "similarity": s["similarity"],
        }
        for s in sources
    ]
    score_breakdown = {
        "retrieval": trust_score.retrieval_similarity,
        "source_count": trust_score.source_count_score,
        "agreement": trust_score.source_agreement,
        "hallucination": 20.0 if trust_score.hallucination_free else max(0, 20.0 - len(trust_score.hallucination_flags) * 5),
    }

    await session.execute(
        text("""
            INSERT INTO audit_logs
                (id, query, answer, confidence_score, confidence_level,
                 sources, hallucination_flags, consistency_check, score_breakdown)
            VALUES
                (:id, :query, :answer, :score, :level,
                 CAST(:sources AS jsonb), CAST(:hall_flags AS jsonb), CAST(:consistency AS jsonb), CAST(:breakdown AS jsonb))
        """),
        {
            "id": audit_id,
            "query": request.question,
            "answer": answer,
            "score": trust_score.score,
            "level": trust_score.level,
            "sources": json.dumps(source_records),
            "hall_flags": json.dumps(trust_score.hallucination_flags),
            "consistency": json.dumps(consistency_result) if consistency_result else None,
            "breakdown": json.dumps(score_breakdown),
        },
    )
    await session.commit()

    # 7. Build response
    return QueryResponse(
        answer=answer,
        confidence=ConfidenceResponse(
            score=trust_score.score,
            level=trust_score.level,
            breakdown=score_breakdown,
        ),
        sources=[
            SourceResponse(
                document=s["filename"],
                page=s["page_number"],
                text=s["content"][:300],
                similarity=round(s["similarity"], 4),
            )
            for s in sources
        ],
        hallucination_check={
            "passed": trust_score.hallucination_free,
            "flags": trust_score.hallucination_flags,
        },
        consistency_check=consistency_result,
        audit_id=audit_id,
    )


@router.post("/demo", response_model=QueryResponse)
async def demo_query():
    """Hardcoded demo response for frontend development without API keys."""
    return QueryResponse(
        answer=(
            "According to OSHA regulation 1926.451, scaffolding must be erected on sound, "
            "rigid footing capable of carrying the maximum intended load without settling or "
            "displacement [Source: OSHA_1926.pdf, p.12]. Workers on scaffolds more than 10 feet "
            "above a lower level must be protected from falling by guardrails or a personal fall "
            "arrest system [Source: OSHA_1926.pdf, p.15]. Additionally, the scaffold platform must "
            "be at least 18 inches wide [Source: Safety_Manual_2024.pdf, p.8]."
        ),
        confidence=ConfidenceResponse(
            score=87.5,
            level="high",
            breakdown={
                "retrieval": 92.0,
                "source_count": 20.0,
                "agreement": 18.5,
                "hallucination": 20.0,
            },
        ),
        sources=[
            SourceResponse(
                document="OSHA_1926.pdf",
                page=12,
                text="Scaffolding must be erected on sound, rigid footing capable of carrying the maximum intended load...",
                similarity=0.9234,
            ),
            SourceResponse(
                document="OSHA_1926.pdf",
                page=15,
                text="Each employee on a scaffold more than 10 feet above a lower level shall be protected from falling...",
                similarity=0.9102,
            ),
            SourceResponse(
                document="Safety_Manual_2024.pdf",
                page=8,
                text="All scaffold platforms shall be at least 18 inches (46 cm) wide...",
                similarity=0.8876,
            ),
        ],
        hallucination_check={"passed": True, "flags": []},
        consistency_check=None,
        audit_id="demo-00000000-0000-0000-0000-000000000000",
    )
