"""Tests for QueryTask async state machine (P1-2) and Groq streaming (P1-3)."""

from unittest.mock import AsyncMock, patch

import pytest

from services.streaming import QueryTask, _GroqRateLimitError


class FakeWebSocket:
    """Minimal WebSocket mock that collects sent frames."""

    def __init__(self):
        self.frames: list[dict] = []

    async def send_json(self, data: dict):
        self.frames.append(data)


@pytest.fixture
def ws():
    return FakeWebSocket()


def _mock_sources():
    return [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "filename": "Safety.pdf",
            "content": "Fall protection is required above 6 feet.",
            "page_number": 3,
            "similarity": 0.92,
        }
    ]


def _mock_trust():
    """Return a mock TrustScore dataclass-like object."""
    from services.trust_verifier import TrustScore
    return TrustScore(
        score=85.0,
        level="high",
        retrieval_similarity=92.0,
        source_count_score=5.0,
        source_agreement=18.0,
        hallucination_free=True,
        hallucination_flags=[],
    )


async def _fake_stream(*args, **kwargs):
    """Fake token generator."""
    tokens = ["Hello", " world", "!"]
    for t in tokens:
        yield t


@pytest.mark.asyncio
async def test_full_pipeline_emits_expected_sequence(ws):
    """Full pipeline emits: status(retrieving) → sources → status(generating) → tokens → status(verifying_trust) → trust → done."""
    task = QueryTask(query_id="q1", text="What is fall protection?")

    with (
        patch.object(task, "_retrieve", new_callable=AsyncMock, return_value=_mock_sources()),
        patch("services.rag_engine.generate_answer_stream", side_effect=_fake_stream),
        patch.object(task, "_verify_trust", new_callable=AsyncMock, return_value={
            "score": 85.0, "level": "high",
            "breakdown": {"retrieval": 92.0, "source_count": 5.0, "agreement": 18.0, "hallucination": 20.0},
        }),
        patch.object(task, "_write_audit", new_callable=AsyncMock, return_value="audit-001"),
    ):
        await task.run(ws)

    types = [f["type"] for f in ws.frames]
    assert types[0] == "status"
    assert ws.frames[0]["stage"] == "retrieving"
    assert "sources" in types
    assert types.count("token") == 3
    assert "trust" in types
    assert types[-1] == "done"
    assert ws.frames[-1]["audit_id"] == "audit-001"


@pytest.mark.asyncio
async def test_cancel_during_retrieval(ws):
    """Cancel set before retrieve returns → emits cancelled, no tokens."""
    task = QueryTask(query_id="q2", text="test")
    task.cancelled.set()  # pre-cancel

    with (
        patch.object(task, "_retrieve", new_callable=AsyncMock, return_value=_mock_sources()),
        patch.object(task, "_write_partial_audit", new_callable=AsyncMock),
    ):
        await task.run(ws)

    types = [f["type"] for f in ws.frames]
    assert "cancelled" in types
    assert "token" not in types


@pytest.mark.asyncio
async def test_cancel_during_generation(ws):
    """Cancel after some tokens → emits cancelled with partial answer preserved."""
    task = QueryTask(query_id="q3", text="test")

    token_count = 0

    async def _slow_stream(*args, **kwargs):
        nonlocal token_count
        for t in ["A", "B", "C", "D", "E"]:
            token_count += 1
            if token_count == 3:
                task.cancelled.set()
            yield t

    with (
        patch.object(task, "_retrieve", new_callable=AsyncMock, return_value=_mock_sources()),
        patch("services.rag_engine.generate_answer_stream", side_effect=_slow_stream),
        patch.object(task, "_write_partial_audit", new_callable=AsyncMock) as mock_audit,
    ):
        await task.run(ws)

    types = [f["type"] for f in ws.frames]
    assert "cancelled" in types
    # Should have emitted some tokens before cancel
    token_frames = [f for f in ws.frames if f["type"] == "token"]
    assert len(token_frames) >= 2
    # Partial answer accumulated
    assert len(task.partial_answer) > 0
    mock_audit.assert_called_once()


@pytest.mark.asyncio
async def test_exception_emits_error_frame(ws):
    """Exception during pipeline → error frame with INTERNAL code."""
    task = QueryTask(query_id="q4", text="test")

    with patch.object(task, "_retrieve", new_callable=AsyncMock, side_effect=RuntimeError("db down")):
        await task.run(ws)

    types = [f["type"] for f in ws.frames]
    assert "error" in types
    error_frame = next(f for f in ws.frames if f["type"] == "error")
    assert error_frame["code"] == "INTERNAL"
    assert "db down" in error_frame["message"]


@pytest.mark.asyncio
async def test_groq_rate_limit_emits_error_frame(ws):
    """Groq 429 → GROQ_RATE_LIMIT error frame with retry_after_ms."""
    task = QueryTask(query_id="q5", text="test")

    async def _rate_limited_stream(*args, **kwargs):
        yield "partial"
        raise _GroqRateLimitError(retry_after_ms=5000)

    with (
        patch.object(task, "_retrieve", new_callable=AsyncMock, return_value=_mock_sources()),
        patch("services.rag_engine.generate_answer_stream", side_effect=_rate_limited_stream),
    ):
        await task.run(ws)

    types = [f["type"] for f in ws.frames]
    assert "error" in types
    error_frame = next(f for f in ws.frames if f["type"] == "error")
    assert error_frame["code"] == "GROQ_RATE_LIMIT"
    assert error_frame["retry_after_ms"] == 5000


def test_ws_ping_interval_configured():
    """Verify WS_PING_INTERVAL is set to 20 by reading main.py source."""
    from pathlib import Path
    main_src = (Path(__file__).parent.parent / "main.py").read_text(encoding="utf-8")
    assert "WS_PING_INTERVAL = 20" in main_src
    assert "ws_ping_interval=WS_PING_INTERVAL" in main_src
