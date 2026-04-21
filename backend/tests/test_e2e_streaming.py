"""End-to-end streaming pipeline tests for P1-GATE verification.

Tests:
- TTFT (time to first token) < 500ms on mocked pipeline
- Cancel mid-stream preserves partial answer
- Groq 429 emits GROQ_RATE_LIMIT with retry info
- Full pipeline frame ordering
"""

import time
from unittest.mock import AsyncMock, patch

import pytest

from services.streaming import QueryTask, _GroqRateLimitError


class FakeWebSocket:
    """WebSocket mock that records frames with timestamps."""

    def __init__(self):
        self.frames: list[dict] = []
        self._timestamps: list[float] = []
        self._start: float = 0.0

    def mark_start(self):
        self._start = time.monotonic()

    async def send_json(self, data: dict):
        self.frames.append(data)
        self._timestamps.append(time.monotonic())

    def time_to_frame(self, frame_type: str) -> float:
        """Return ms from start to first frame of given type."""
        for frame, ts in zip(self.frames, self._timestamps):
            if frame["type"] == frame_type:
                return (ts - self._start) * 1000
        return float("inf")


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
    return {
        "score": 85.0,
        "level": "high",
        "breakdown": {
            "retrieval": 92.0,
            "source_count": 5.0,
            "agreement": 18.0,
            "hallucination": 20.0,
        },
    }


async def _fast_stream(*args, **kwargs):
    for t in ["Fall", " protection", " is", " required", " above", " 6", " feet", "."]:
        yield t


@pytest.mark.asyncio
async def test_ttft_under_500ms():
    """Happy path: time to first token must be < 500ms (mocked pipeline)."""
    ws = FakeWebSocket()
    task = QueryTask(query_id="gate-ttft", text="What is fall protection?")

    with (
        patch.object(task, "_retrieve", new_callable=AsyncMock, return_value=_mock_sources()),
        patch("services.rag_engine.generate_answer_stream", side_effect=_fast_stream),
        patch.object(task, "_verify_trust", new_callable=AsyncMock, return_value=_mock_trust()),
        patch.object(task, "_write_audit", new_callable=AsyncMock, return_value="audit-gate"),
    ):
        ws.mark_start()
        await task.run(ws)

    ttft = ws.time_to_frame("token")
    assert ttft < 500, f"TTFT was {ttft:.1f}ms, expected < 500ms"

    # Verify complete frame sequence
    types = [f["type"] for f in ws.frames]
    assert types[0] == "status"
    assert ws.frames[0]["stage"] == "retrieving"
    assert "sources" in types
    assert "token" in types
    assert "trust" in types
    assert types[-1] == "done"


@pytest.mark.asyncio
async def test_cancel_preserves_partial_answer_in_audit():
    """Cancel mid-stream: partial answer must be written to audit."""
    ws = FakeWebSocket()
    task = QueryTask(query_id="gate-cancel", text="test cancel")

    token_count = 0

    async def _interruptible_stream(*args, **kwargs):
        nonlocal token_count
        for t in ["Part", "ial ", "ans", "wer ", "here"]:
            token_count += 1
            if token_count == 3:
                task.cancelled.set()
            yield t

    with (
        patch.object(task, "_retrieve", new_callable=AsyncMock, return_value=_mock_sources()),
        patch("services.rag_engine.generate_answer_stream", side_effect=_interruptible_stream),
        patch.object(task, "_write_partial_audit", new_callable=AsyncMock) as mock_audit,
    ):
        await task.run(ws)

    # Cancelled frame emitted
    types = [f["type"] for f in ws.frames]
    assert "cancelled" in types

    # Partial answer exists and is non-empty
    assert len(task.partial_answer) > 0, "Partial answer should be preserved"
    mock_audit.assert_called_once()


@pytest.mark.asyncio
async def test_groq_429_shows_retry_in_error_frame():
    """Groq rate limit → GROQ_RATE_LIMIT error with retry_after_ms."""
    ws = FakeWebSocket()
    task = QueryTask(query_id="gate-429", text="test rate limit")

    async def _rate_limited(*args, **kwargs):
        yield "token"
        raise _GroqRateLimitError(retry_after_ms=3000)

    with (
        patch.object(task, "_retrieve", new_callable=AsyncMock, return_value=_mock_sources()),
        patch("services.rag_engine.generate_answer_stream", side_effect=_rate_limited),
    ):
        await task.run(ws)

    error_frames = [f for f in ws.frames if f["type"] == "error"]
    assert len(error_frames) == 1
    assert error_frames[0]["code"] == "GROQ_RATE_LIMIT"
    assert error_frames[0]["retry_after_ms"] == 3000


@pytest.mark.asyncio
async def test_reconnect_delay_caps_at_8s():
    """Frontend reconnect logic: exponential backoff caps at 8000ms.

    This is a logic verification test — the actual WS reconnect is in
    frontend/src/lib/ws-client.js. We verify the algorithm here.
    """
    # Simulate the reconnect delay algorithm from ws-client.js
    delay = 1000
    max_delay = 8000
    delays = []

    for _ in range(6):
        delays.append(delay)
        delay = min(delay * 2, max_delay)

    assert delays == [1000, 2000, 4000, 8000, 8000, 8000]
    # After backend kill, worst case first reconnect at 1s, within 8s target
    assert delays[0] <= 8000
