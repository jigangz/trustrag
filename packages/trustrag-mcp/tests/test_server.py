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
    mock_client.query = AsyncMock(return_value={
        "answer": "The threshold is 6 feet.",
        "trust_score": 85,
        "sources": [
            {"doc": "osha_manual.pdf", "page": 12, "similarity": 0.92},
        ],
        "trust_breakdown": {"faithfulness": 0.9, "relevance": 0.8},
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
        "trust_score": 60,
        "sources": [],
        "trust_breakdown": {},
    })
    result = await call_tool(
        "trustrag_query", {"question": "test", "min_trust_score": 70}
    )
    assert len(result) == 1
    assert "No trustworthy answer" in result[0].text
    assert "60" in result[0].text
    assert "70" in result[0].text


@pytest.mark.asyncio
async def test_call_upload_success(mock_client):
    mock_client.upload_document = AsyncMock(return_value={
        "id": "doc-123",
        "num_chunks": 42,
    })
    result = await call_tool(
        "trustrag_upload_document", {"file_path": "/tmp/test.pdf"}
    )
    assert len(result) == 1
    assert "doc-123" in result[0].text
    assert "42 chunks" in result[0].text


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
