"""Tests for TrustRAGTool."""

import httpx
import pytest
import respx

from trustrag_langchain.tool import TrustRAGTool


ENDPOINT = "http://testserver:8000"


@pytest.fixture
def tool():
    from trustrag_langchain.retriever import TrustRAGRetriever

    retriever = TrustRAGRetriever(endpoint=ENDPOINT, min_trust_score=70)
    return TrustRAGTool(retriever=retriever)


def test_tool_returns_formatted_answer_with_trust(
    tool, mock_backend_high_trust, respx_mock
):
    """Tool returns string with [Trust: X/100] prefix."""
    respx_mock.post(f"{ENDPOINT}/api/query").mock(
        return_value=httpx.Response(200, json=mock_backend_high_trust)
    )

    result = tool.invoke({"query": "What is fall protection?"})

    assert "[Trust: 85/100]" in result
    assert "fall protection" in result.lower()
    assert "Sources:" in result


def test_tool_returns_no_answer_when_filtered(
    tool, mock_backend_low_trust, respx_mock
):
    """When trust is below threshold, returns helpful 'no answer' message."""
    respx_mock.post(f"{ENDPOINT}/api/query").mock(
        return_value=httpx.Response(200, json=mock_backend_low_trust)
    )

    result = tool.invoke({"query": "vague question"})

    assert "No trustworthy answer" in result


def test_tool_description_mentions_trust(tool):
    """Tool description contains 'trust' for LLM parseability."""
    assert "trust" in tool.description.lower()
