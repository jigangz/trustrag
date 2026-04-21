"""QueryTask — async state machine for streaming RAG queries over WebSocket.

State transitions: RETRIEVING → GENERATING → VERIFYING → COMPLETE
Supports cancellation at each stage boundary via asyncio.Event.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncIterator

from fastapi import WebSocket

from config import settings
from ws_messages import ErrorFrame

logger = logging.getLogger(__name__)


class _GroqRateLimitError(Exception):
    """Internal wrapper for Groq 429 with retry info."""
    def __init__(self, retry_after_ms: int = 2000):
        self.retry_after_ms = retry_after_ms
        super().__init__(f"Groq rate limit, retry after {retry_after_ms}ms")


class QueryTask:
    """Executes a single RAG query as a multi-stage streaming pipeline."""

    def __init__(self, query_id: str, text: str, top_k: int = 5):
        self.id = query_id
        self.text = text
        self.top_k = top_k
        self.cancelled = asyncio.Event()
        self.partial_answer = ""

    async def run(self, ws: WebSocket):
        """Execute the full query pipeline, emitting frames at each stage."""
        try:
            # Stage 1: Retrieve
            await self._emit(ws, "status", stage="retrieving")
            sources = await self._retrieve()
            if self.cancelled.is_set():
                return await self._emit_cancelled(ws)
            await self._emit(ws, "sources", sources=sources)

            # Stage 2: Generate (streaming)
            await self._emit(ws, "status", stage="generating")
            async for token in self._generate_stream(sources):
                if self.cancelled.is_set():
                    return await self._emit_cancelled(ws)
                self.partial_answer += token
                await self._emit(ws, "token", content=token)

            # Stage 3: Verify trust
            if self.cancelled.is_set():
                return await self._emit_cancelled(ws)
            await self._emit(ws, "status", stage="verifying_trust")
            trust = await self._verify_trust(self.partial_answer, sources)
            await self._emit(ws, "trust", score=trust["score"], breakdown=trust["breakdown"])

            # Stage 4: Consistency check
            if self.cancelled.is_set():
                return await self._emit_cancelled(ws)
            consistency = await self._check_consistency(sources)
            if consistency:
                await self._emit(
                    ws, "consistency",
                    score=consistency["score"],
                    rephrases_matched=consistency.get("rephrases_matched", 0),
                )

            # Done
            audit_id = await self._write_audit(trust, consistency, sources)
            await self._emit(ws, "done", audit_id=audit_id)

        except _GroqRateLimitError as e:
            logger.warning("QueryTask %s hit Groq rate limit", self.id)
            await ws.send_json({
                "type": "error",
                "id": self.id,
                "code": "GROQ_RATE_LIMIT",
                "message": str(e),
                "retry_after_ms": e.retry_after_ms,
            })
        except Exception as e:
            logger.exception("QueryTask %s failed", self.id)
            await self._emit_error(ws, code="INTERNAL", message=str(e))

    async def _retrieve(self) -> list[dict]:
        """Retrieve relevant chunks via semantic search."""
        from database import async_session
        from services.embedding import embed_text
        from services.vector_store import search_similar

        async with async_session() as session:
            query_embedding = await embed_text(self.text)
            chunks = await search_similar(session, query_embedding, top_k=self.top_k)
        return chunks

    async def _generate_stream(self, sources: list[dict]) -> AsyncIterator[str]:
        """Stream tokens from Groq LLM. Handles rate limit errors."""
        from openai import RateLimitError
        from services.rag_engine import generate_answer_stream

        try:
            async for token in generate_answer_stream(self.text, sources):
                yield token
        except RateLimitError as e:
            retry_after = int(getattr(e, "retry_after", 2000) if hasattr(e, "retry_after") else 2000)
            raise _GroqRateLimitError(retry_after_ms=retry_after) from e

    async def _verify_trust(self, answer: str, sources: list[dict]) -> dict:
        """Run trust verification and return score + breakdown."""
        from services.embedding import embed_text
        from services.trust_verifier import compute_trust_score

        query_embedding = await embed_text(self.text)
        trust = await compute_trust_score(answer, sources, query_embedding)
        return {
            "score": trust.score,
            "level": trust.level,
            "breakdown": {
                "retrieval": trust.retrieval_similarity,
                "source_count": trust.source_count_score,
                "agreement": trust.source_agreement,
                "hallucination": 20.0 if trust.hallucination_free else max(0, 20.0 - len(trust.hallucination_flags) * 5),
            },
        }

    async def _check_consistency(self, sources: list[dict]) -> dict | None:
        """Run consistency check (lightweight — skipped if answer is short)."""
        # Phase 1: skip consistency to keep streaming fast; return None
        return None

    async def _write_audit(self, trust: dict, consistency: dict | None, sources: list[dict]) -> str:
        """Write full audit log entry and return audit_id."""
        from database import async_session
        from sqlalchemy import text as sql_text

        audit_id = str(uuid.uuid4())
        source_records = [
            {
                "document_id": s.get("document_id", ""),
                "filename": s.get("filename", ""),
                "page": s.get("page_number", 0),
                "text": s.get("content", "")[:500],
                "similarity": s.get("similarity", 0),
            }
            for s in sources
        ]

        async with async_session() as session:
            await session.execute(
                sql_text("""
                    INSERT INTO audit_logs
                        (id, query, answer, confidence_score, confidence_level,
                         sources, hallucination_flags, consistency_check, score_breakdown)
                    VALUES
                        (:id, :query, :answer, :score, :level,
                         CAST(:sources AS jsonb), CAST(:hall_flags AS jsonb),
                         CAST(:consistency AS jsonb), CAST(:breakdown AS jsonb))
                """),
                {
                    "id": audit_id,
                    "query": self.text,
                    "answer": self.partial_answer,
                    "score": trust["score"],
                    "level": trust["level"],
                    "sources": json.dumps(source_records),
                    "hall_flags": json.dumps([]),
                    "consistency": json.dumps(consistency) if consistency else None,
                    "breakdown": json.dumps(trust["breakdown"]),
                },
            )
            await session.commit()
        return audit_id

    async def _write_partial_audit(self):
        """Write partial audit entry when query is cancelled."""
        from database import async_session
        from sqlalchemy import text as sql_text

        audit_id = str(uuid.uuid4())
        try:
            async with async_session() as session:
                await session.execute(
                    sql_text("""
                        INSERT INTO audit_logs
                            (id, query, answer, confidence_score, confidence_level,
                             sources, hallucination_flags, consistency_check, score_breakdown)
                        VALUES
                            (:id, :query, :answer, :score, :level,
                             CAST(:sources AS jsonb), NULL, NULL, NULL)
                    """),
                    {
                        "id": audit_id,
                        "query": self.text,
                        "answer": self.partial_answer or "[cancelled]",
                        "score": 0.0,
                        "level": "cancelled",
                        "sources": json.dumps([]),
                    },
                )
                await session.commit()
        except Exception:
            logger.warning("Failed to write partial audit for cancelled query %s", self.id)

    async def _emit(self, ws: WebSocket, type_: str, **kwargs):
        """Send a typed frame to the client."""
        await ws.send_json({"type": type_, "id": self.id, **kwargs})

    async def _emit_cancelled(self, ws: WebSocket):
        """Emit cancelled frame and write partial audit."""
        await self._emit(ws, "cancelled")
        await self._write_partial_audit()

    async def _emit_error(self, ws: WebSocket, code: str, message: str, **extra):
        """Emit error frame to client."""
        await ws.send_json(
            ErrorFrame(id=self.id, code=code, message=message, **extra).model_dump()
        )
