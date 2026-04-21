"""Tests for TrustRAGRetriever."""

import httpx
import pytest
import respx

from trustrag_langchain.retriever import TrustRAGRetriever


ENDPOINT = "http://testserver:8000"


@pytest.fixture
def retriever():
    return TrustRAGRetriever(endpoint=ENDPOINT, min_trust_score=70)


def test_retriever_returns_document_above_threshold(
    retriever, mock_backend_high_trust, respx_mock
):
    """Trust score 85 >= threshold 70 → returns 1 document."""
    respx_mock.post(f"{ENDPOINT}/api/query").mock(
        return_value=httpx.Response(200, json=mock_backend_high_trust)
    )

    docs = retriever.invoke("What is fall protection?")

    assert len(docs) == 1
    assert "fall protection" in docs[0].page_content.lower()


def test_retriever_filters_below_threshold(
    retriever, mock_backend_low_trust, respx_mock
):
    """Trust score 45 < threshold 70 → returns empty list (not exception)."""
    respx_mock.post(f"{ENDPOINT}/api/query").mock(
        return_value=httpx.Response(200, json=mock_backend_low_trust)
    )

    docs = retriever.invoke("vague question")

    assert docs == []


def test_retriever_metadata_includes_trust_score(
    retriever, mock_backend_high_trust, respx_mock
):
    """Metadata contains trust_score, sources, and audit_id."""
    respx_mock.post(f"{ENDPOINT}/api/query").mock(
        return_value=httpx.Response(200, json=mock_backend_high_trust)
    )

    docs = retriever.invoke("question")

    assert docs[0].metadata["trust_score"] == 85
    assert len(docs[0].metadata["sources"]) == 2
    assert docs[0].metadata["audit_id"] == "aud_abc123"


@pytest.mark.asyncio
async def test_async_retriever_works(
    retriever, mock_backend_high_trust, respx_mock
):
    """Async path returns same results as sync."""
    respx_mock.post(f"{ENDPOINT}/api/query").mock(
        return_value=httpx.Response(200, json=mock_backend_high_trust)
    )

    docs = await retriever.ainvoke("async question")

    assert len(docs) == 1
    assert docs[0].metadata["trust_score"] == 85


def test_retriever_handles_backend_500(retriever, respx_mock):
    """Backend 500 raises httpx.HTTPStatusError (not swallowed)."""
    respx_mock.post(f"{ENDPOINT}/api/query").mock(
        return_value=httpx.Response(500, json={"error": "internal"})
    )

    with pytest.raises(httpx.HTTPStatusError):
        retriever.invoke("question")
