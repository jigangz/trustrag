"""Tests for merged generation+self-check prompt (Fix 3, OPT-3)."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock fastembed before importing modules that depend on it
sys.modules.setdefault("fastembed", MagicMock())


def _make_groq_response(content: str):
    """Build a fake Groq response matching OpenAI client shape."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.model_dump.return_value = {"id": "test"}
    return resp


@pytest.mark.asyncio
async def test_merged_happy_path_returns_answer_and_flags():
    from services.rag_engine import generate_answer_merged

    chunks = [
        {"filename": "osha.pdf", "page_number": 12, "content": "Fall protection required at 6 feet."},
    ]

    fake_json = json.dumps({
        "answer": "OSHA requires fall protection at 6 feet [Source: osha.pdf, p.12].",
        "self_check": {
            "unsupported_claims": []
        }
    })

    with patch("services.rag_engine.client.chat.completions.create",
               new=AsyncMock(return_value=_make_groq_response(fake_json))):
        result = await generate_answer_merged("What height requires fall protection?", chunks)

    assert "6 feet" in result["answer"]
    assert result["hallucination_flags"] == []
    assert result["merged"] is True
    assert any(c["document"] == "osha.pdf" for c in result["sources_used"])


@pytest.mark.asyncio
async def test_merged_with_unsupported_claims():
    from services.rag_engine import generate_answer_merged

    chunks = [{"filename": "osha.pdf", "page_number": 12, "content": "Fall protection required."}]

    fake_json = json.dumps({
        "answer": "OSHA requires fall protection at 6 feet [Source: osha.pdf, p.12]. Helmets must be blue.",
        "self_check": {
            "unsupported_claims": [
                {"sentence": "Helmets must be blue.", "reason": "Sources do not specify helmet color."}
            ]
        }
    })

    with patch("services.rag_engine.client.chat.completions.create",
               new=AsyncMock(return_value=_make_groq_response(fake_json))):
        result = await generate_answer_merged("test", chunks)

    assert len(result["hallucination_flags"]) == 1
    assert "blue" in result["hallucination_flags"][0]["sentence"].lower()
    assert result["merged"] is True


@pytest.mark.asyncio
async def test_merged_fallback_on_invalid_json():
    """If Groq returns non-JSON, fall back to 2-call path."""
    from services.rag_engine import generate_answer_merged

    chunks = [{"filename": "osha.pdf", "page_number": 12, "content": "Fall protection text."}]

    # First call (merged) returns broken JSON
    broken_resp = _make_groq_response("This is not JSON at all")
    # Fallback call (generate_answer) returns normal answer
    fallback_resp = _make_groq_response("OSHA requires fall protection [Source: osha.pdf, p.12].")

    call_count = {"n": 0}

    async def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return broken_resp
        return fallback_resp

    with patch("services.rag_engine.client.chat.completions.create", new=fake_create), \
         patch("services.trust_verifier._check_hallucination", new=AsyncMock(return_value=[])):
        result = await generate_answer_merged("test", chunks)

    assert result["merged"] is False, "Expected fallback path"
    assert "OSHA" in result["answer"]
    assert result["hallucination_flags"] == []
