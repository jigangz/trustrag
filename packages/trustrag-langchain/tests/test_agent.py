"""Tests for TrustBudgetAgent (L3 LangGraph agent)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from trustrag_langchain.agent import TrustBudgetAgent
from trustrag_langchain.retriever import TrustRAGRetriever


def _make_retriever_mock(responses: list[list[Document]]) -> TrustRAGRetriever:
    """Create a mock retriever that returns documents from a list in sequence."""
    retriever = MagicMock(spec=TrustRAGRetriever)
    retriever.ainvoke = AsyncMock(side_effect=responses)
    return retriever


def _make_llm_mock(responses: list[str]) -> MagicMock:
    """Create a mock LLM that returns AIMessage responses in sequence."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        side_effect=[AIMessage(content=r) for r in responses]
    )
    return llm


def _doc(answer: str, trust: int, sources: list[dict] | None = None) -> Document:
    return Document(
        page_content=answer,
        metadata={
            "trust_score": trust,
            "sources": sources or [{"doc": "test.pdf", "page": 1}],
            "audit_id": "aud_test",
        },
    )


@pytest.mark.asyncio
async def test_agent_answers_when_cumulative_trust_meets_threshold():
    """High trust on first retrieval + low threshold → immediate answer."""
    retriever = _make_retriever_mock([
        [_doc("Fall protection required above 6ft.", 90)],
    ])
    llm = _make_llm_mock(["ANSWER"])

    agent = TrustBudgetAgent(
        retriever=retriever,
        llm=llm,
        min_trust_threshold=80,  # Met on first retrieval (90 >= 80)
        max_retrievals=3,
    )

    result = await agent.ainvoke("What is fall protection?")

    assert result["outcome"] == "answer"
    assert result["cumulative_trust"] == 90
    assert "Fall protection" in result["answer"]
    assert len(result["retrievals"]) == 1


@pytest.mark.asyncio
async def test_agent_stops_at_max_retrievals_below_threshold():
    """All retrievals low trust, max reached → stop_low_trust."""
    retriever = _make_retriever_mock([
        [_doc("Maybe harnesses", 30)],
        [_doc("Something about ropes", 30)],
        [_doc("Unclear guidance", 30)],
    ])
    # LLM asks subqueries since trust is low
    llm = _make_llm_mock([
        "SUBQUERY: What specific PPE for heights?",
        "SUBQUERY: Are harnesses mandatory?",
    ])

    agent = TrustBudgetAgent(
        retriever=retriever,
        llm=llm,
        min_trust_threshold=150,
        max_retrievals=3,
    )

    result = await agent.ainvoke("What PPE is required?")

    assert result["outcome"] == "stop_low_trust"
    assert result["cumulative_trust"] == 90  # 30 * 3
    assert "Insufficient evidence" in result["answer"]
    assert len(result["retrievals"]) == 3


@pytest.mark.asyncio
async def test_agent_continues_with_subquery():
    """Agent asks a sub-question and retrieves again on second pass."""
    retriever = _make_retriever_mock([
        [_doc("Partial info about harnesses", 60)],
        [_doc("Complete PPE requirements for heights", 95)],
    ])
    # First decide: not enough, ask subquery. Second: threshold met (60+95=155 >= 150)
    llm = _make_llm_mock(["SUBQUERY: What specific harness standards apply?"])

    agent = TrustBudgetAgent(
        retriever=retriever,
        llm=llm,
        min_trust_threshold=150,
        max_retrievals=3,
    )

    result = await agent.ainvoke("What PPE is needed?")

    assert result["outcome"] == "answer"
    assert result["cumulative_trust"] == 155  # 60 + 95
    assert len(result["retrievals"]) == 2
    # Should have used the subquery
    assert retriever.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_agent_synthesis_excludes_low_trust():
    """Synthesis only includes retrievals with trust >= 50."""
    retriever = _make_retriever_mock([
        [_doc("High trust answer about PPE", 90, [{"doc": "osha.pdf", "page": 1}])],
        [_doc("Low trust garbage", 30, [{"doc": "bad.pdf", "page": 2}])],
        [_doc("Another good answer", 80, [{"doc": "safety.pdf", "page": 3}])],
    ])
    llm = _make_llm_mock([
        "SUBQUERY: More details?",
        "SUBQUERY: Any other requirements?",
    ])

    agent = TrustBudgetAgent(
        retriever=retriever,
        llm=llm,
        min_trust_threshold=250,  # Won't be met (90+30+80=200 < 250)
        max_retrievals=3,
    )

    # Since cumulative 200 < 250 and max_retrievals hit, this will stop_low_trust.
    # But let's test synthesis separately by setting threshold low enough to answer.
    agent2 = TrustBudgetAgent(
        retriever=_make_retriever_mock([
            [_doc("High trust answer about PPE", 90, [{"doc": "osha.pdf", "page": 1}])],
            [_doc("Low trust garbage", 30, [{"doc": "bad.pdf", "page": 2}])],
            [_doc("Another good answer", 80, [{"doc": "safety.pdf", "page": 3}])],
        ]),
        llm=_make_llm_mock([
            "SUBQUERY: More details?",
            "SUBQUERY: Any other?",
            "ANSWER",
        ]),
        min_trust_threshold=190,  # Met at 90+30+80=200 >= 190 after 3rd retrieval
        max_retrievals=5,
    )

    result = await agent2.ainvoke("PPE requirements?")

    assert result["outcome"] == "answer"
    # Synthesis should exclude the 30-trust retrieval
    assert "Low trust garbage" not in result["answer"]
    assert "High trust answer" in result["answer"]
    assert "Another good answer" in result["answer"]


@pytest.mark.asyncio
async def test_agent_budget_exhausted_emits_stop_outcome():
    """Verify outcome field is exactly 'stop_low_trust' when budget exhausted."""
    retriever = _make_retriever_mock([
        [],  # Empty (filtered by retriever)
        [],
        [],
    ])
    llm = _make_llm_mock([
        "SUBQUERY: Try another angle",
        "SUBQUERY: Last attempt",
    ])

    agent = TrustBudgetAgent(
        retriever=retriever,
        llm=llm,
        min_trust_threshold=150,
        max_retrievals=3,
    )

    result = await agent.ainvoke("Obscure question?")

    assert result["outcome"] == "stop_low_trust"
    assert result["cumulative_trust"] == 0
    assert "Insufficient evidence" in result["answer"]
    assert "150" in result["answer"]  # Shows the threshold in message
