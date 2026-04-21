"""TrustRAG LangChain BaseTool wrapper for agent usage."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from trustrag_langchain.retriever import TrustRAGRetriever


class TrustRAGToolInput(BaseModel):
    query: str = Field(description="The question to ask the TrustRAG knowledge base")


class TrustRAGTool(BaseTool):
    """Query a trust-verified RAG knowledge base.

    Returns an answer prefixed with [Trust: X/100]. Answers below the
    trust threshold are filtered out; empty result means no trustworthy
    answer available.
    """

    name: str = "trustrag_query"
    description: str = (
        "Query a trust-verified RAG knowledge base. Returns an answer with "
        "trust score (0-100). Answers below the trust threshold are filtered "
        "out; empty result means no trustworthy answer available. Prefer this "
        "tool when factual accuracy matters."
    )
    args_schema: type[BaseModel] = TrustRAGToolInput

    retriever: TrustRAGRetriever = Field(default_factory=TrustRAGRetriever)

    def _run(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        if not docs:
            return (
                "No trustworthy answer available in knowledge base "
                "(all retrieved answers below trust threshold)."
            )
        doc = docs[0]
        trust = doc.metadata["trust_score"]
        sources = doc.metadata["sources"]
        source_str = "; ".join(
            f"{s['doc']} p.{s.get('page', '?')}" for s in sources[:3]
        )
        return f"[Trust: {trust}/100] {doc.page_content}\n\nSources: {source_str}"

    async def _arun(self, query: str) -> str:
        docs = await self.retriever.ainvoke(query)
        if not docs:
            return (
                "No trustworthy answer available in knowledge base "
                "(all retrieved answers below trust threshold)."
            )
        doc = docs[0]
        trust = doc.metadata["trust_score"]
        sources = doc.metadata["sources"]
        source_str = "; ".join(
            f"{s['doc']} p.{s.get('page', '?')}" for s in sources[:3]
        )
        return f"[Trust: {trust}/100] {doc.page_content}\n\nSources: {source_str}"
