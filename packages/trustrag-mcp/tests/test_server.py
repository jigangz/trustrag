"""Tests for the TrustRAG MCP server."""

import pytest
from unittest.mock import AsyncMock, patch

from trustrag_mcp.server import list_tools, call_tool


@pytest.fixture
def mock_client():
    with patch("trustrag_mcp.server.client") as m:
        yield m


@pytest.mark.asyncio
async def test_list_tools_returns_three():
    tools = await list_tools()
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {"trustrag_query", "trustrag_upload_document", "trustrag_get_audit_log"}


@pytest.mark.asyncio
async def test_tool_schemas_valid():
    """Each inputSchema must be a valid JSON Schema object with type and properties."""
    tools = await list_tools()
    for tool in tools:
        schema = tool.inputSchema
        assert schema["type"] == "object"
        assert "properties" in schema


@pytest.mark.asyncio
async def test_call_query_formats_response(mock_client):
    # Real backend QueryResponse shape: confidence.{score,level,breakdown}, sources[*].document
    mock_client.query = AsyncMock(return_value={
        "answer": "The threshold is 6 feet.",
        "confidence": {
            "score": 85,
            "level": "high",
            "breakdown": {"faithfulness": 0.9, "relevance": 0.8},
        },
        "sources": [
            {"document": "osha_manual.pdf", "page": 12, "similarity": 0.92, "text": "..."},
        ],
        "audit_id": "abc-123",
    })
    result = await call_tool("trustrag_query", {"question": "What is the height threshold?"})
    assert len(result) == 1
    assert "Trust: 85/100" in result[0].text
    assert "The threshold is 6 feet." in result[0].text
    assert "osha_manual.pdf" in result[0].text


@pytest.mark.asyncio
async def test_call_query_respects_min_trust_score(mock_client):
    mock_client.query = AsyncMock(return_value={
        "answer": "Some answer",
        "confidence": {"score": 60, "level": "low", "breakdown": {}},
        "sources": [],
    })
    result = await call_tool(
        "trustrag_query", {"question": "test", "min_trust_score": 70}
    )
    assert len(result) == 1
    assert "No trustworthy answer" in result[0].text
    assert "60" in result[0].text
    assert "70" in result[0].text


@pytest.mark.asyncio
async def test_call_query_legacy_shape_still_works(mock_client):
    """Backwards compat: if backend ever returns flat trust_score (older deploy), still parse."""
    mock_client.query = AsyncMock(return_value={
        "answer": "Legacy shape answer.",
        "trust_score": 77,
        "sources": [{"doc": "old.pdf", "page": 1, "similarity": 0.5}],
        "trust_breakdown": {},
    })
    result = await call_tool("trustrag_query", {"question": "test"})
    assert "Trust: 77/100" in result[0].text
    assert "old.pdf" in result[0].text


@pytest.mark.asyncio
async def test_call_upload_success(mock_client):
    # Real backend DocumentResponse uses 'total_chunks'
    mock_client.upload_document = AsyncMock(return_value={
        "id": "doc-123",
        "filename": "test.pdf",
        "uploaded_at": "2026-04-25T00:00:00Z",
        "total_pages": 2,
        "total_chunks": 42,
    })
    result = await call_tool(
        "trustrag_upload_document", {"file_path": "/tmp/test.pdf"}
    )
    assert len(result) == 1
    assert "doc-123" in result[0].text
    assert "42 chunks" in result[0].text


@pytest.mark.asyncio
async def test_call_audit_formats_real_shape(mock_client):
    # Real backend audit entry shape: query, confidence_score (not question/trust_score)
    mock_client.get_audit_log = AsyncMock(return_value=[
        {
            "id": "a1",
            "query": "What is OSHA?",
            "answer": "OSHA is the Occupational Safety and Health Administration." + " " * 200,
            "confidence_score": 88,
            "confidence_level": "high",
            "created_at": "2026-04-25T10:00:00Z",
        },
    ])
    result = await call_tool("trustrag_get_audit_log", {"limit": 5})
    assert len(result) == 1
    assert "Trust: 88" in result[0].text
    assert "What is OSHA?" in result[0].text
    assert "2026-04-25T10:00:00Z" in result[0].text


@pytest.mark.asyncio
async def test_call_audit_max_trust_filter_clientside(mock_client):
    # max_trust_score is now applied client-side (backend doesn't support it)
    mock_client.get_audit_log = AsyncMock(return_value=[
        {"id": "1", "query": "high", "answer": "a", "confidence_score": 95, "created_at": None},
        {"id": "2", "query": "low",  "answer": "b", "confidence_score": 40, "created_at": None},
    ])
    result = await call_tool(
        "trustrag_get_audit_log", {"limit": 10, "max_trust_score": 80}
    )
    text = result[0].text
    assert "low" in text       # score 40 below 80 — kept
    assert "high" not in text  # score 95 above 80 — filtered out


@pytest.mark.asyncio
async def test_call_tool_unknown_returns_error():
    result = await call_tool("nonexistent_tool", {})
    assert len(result) == 1
    assert "Unknown tool: nonexistent_tool" in result[0].text


@pytest.mark.asyncio
async def test_call_tool_backend_down_returns_error(mock_client):
    mock_client.query = AsyncMock(
        side_effect=ConnectionError("Connection refused")
    )
    result = await call_tool("trustrag_query", {"question": "test"})
    assert len(result) == 1
    assert "Error:" in result[0].text
    assert "ConnectionError" in result[0].text
