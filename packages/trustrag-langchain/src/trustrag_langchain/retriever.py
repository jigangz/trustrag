"""TrustRAG LangChain BaseRetriever with trust score filtering."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class TrustRAGRetriever(BaseRetriever):
    """LangChain BaseRetriever backed by a TrustRAG backend.

    Filters out answers below `min_trust_score` — returns empty list
    instead of raising, keeping agent context clean of untrusted data.
    """

    endpoint: str = "http://localhost:8000"
    min_trust_score: int = Field(default=70, ge=0, le=100)
    top_k: int = Field(default=5, ge=1, le=20)
    timeout: float = 30.0

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.endpoint}/api/query",
                json={"question": query, "top_k": self.top_k},
            )
            resp.raise_for_status()
            data = resp.json()

        if data["trust_score"] < self.min_trust_score:
            return []

        return [
            Document(
                page_content=data["answer"],
                metadata={
                    "trust_score": data["trust_score"],
                    "sources": data["sources"],
                    "audit_id": data.get("audit_id"),
                },
            )
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.endpoint}/api/query",
                json={"question": query, "top_k": self.top_k},
            )
            resp.raise_for_status()
            data = resp.json()

        if data["trust_score"] < self.min_trust_score:
            return []

        return [
            Document(
                page_content=data["answer"],
                metadata={
                    "trust_score": data["trust_score"],
                    "sources": data["sources"],
                    "audit_id": data.get("audit_id"),
                },
            )
        ]
