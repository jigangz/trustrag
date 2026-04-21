"""Shared fixtures for trustrag-langchain tests."""

import pytest
import httpx
import respx


@pytest.fixture
def mock_backend_high_trust():
    """Mock TrustRAG backend returning high trust response."""
    return {
        "answer": "Fall protection is required at heights above 6 feet.",
        "trust_score": 85,
        "sources": [
            {"doc": "OSHA_29CFR1926.pdf", "page": 42},
            {"doc": "safety_manual.pdf", "page": 7},
        ],
        "audit_id": "aud_abc123",
    }


@pytest.fixture
def mock_backend_low_trust():
    """Mock TrustRAG backend returning low trust response."""
    return {
        "answer": "Maybe something about harnesses.",
        "trust_score": 45,
        "sources": [{"doc": "unknown.pdf", "page": 1}],
        "audit_id": "aud_low456",
    }


@pytest.fixture
def respx_mock():
    """Provides a respx mock router for httpx."""
    with respx.mock(assert_all_called=False) as router:
        yield router
