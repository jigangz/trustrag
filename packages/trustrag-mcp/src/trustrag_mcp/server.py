"""MCP server exposing TrustRAG tools over stdio."""

import os

from mcp.server import Server
from mcp.types import Tool, TextContent

from trustrag_mcp.client import TrustRAGClient

BACKEND_URL = os.getenv("TRUSTRAG_BACKEND_URL", "http://localhost:8000")
client = TrustRAGClient(base_url=BACKEND_URL)

app = Server("trustrag-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="trustrag_query",
            description=(
                "Ask the TrustRAG knowledge base a question. Returns an answer with a trust "
                "score (0-100), source citations, and hallucination check result. Answers below "
                "min_trust_score are filtered out (empty response if all below threshold)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask"},
                    "min_trust_score": {
                        "type": "integer",
                        "description": "Filter threshold (0-100)",
                        "default": 0,
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="trustrag_upload_document",
            description=(
                "Upload a PDF document to the TrustRAG knowledge base. The document is parsed, "
                "chunked, and embedded. Returns document_id after processing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to PDF file on local disk",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata tags",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="trustrag_get_audit_log",
            description=(
                "Fetch recent query audit entries from TrustRAG. Useful for reviewing queries "
                "with low trust scores or investigating past answers. Returns list of entries "
                "with question, answer, trust score, sources, and timestamp."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10, "maximum": 100},
                    "max_trust_score": {
                        "type": "integer",
                        "description": "Only return queries with trust BELOW this threshold",
                    },
                    "since_hours": {
                        "type": "integer",
                        "description": "Only entries from last N hours",
                    },
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "trustrag_query":
            return await _query(**arguments)
        elif name == "trustrag_upload_document":
            return await _upload(**arguments)
        elif name == "trustrag_get_audit_log":
            return await _audit(**arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


async def _query(
    question: str, min_trust_score: int = 0, top_k: int = 5
) -> list[TextContent]:
    data = await client.query(question, top_k=top_k)
    if data["trust_score"] < min_trust_score:
        return [
            TextContent(
                type="text",
                text=(
                    f"No trustworthy answer available. "
                    f"Best match had trust score {data['trust_score']}, "
                    f"below threshold {min_trust_score}."
                ),
            )
        ]
    sources_str = "\n".join(
        f"- {s['doc']} (page {s.get('page', '?')}, similarity {s.get('similarity', 0):.2f})"
        for s in data["sources"][:5]
    )
    return [
        TextContent(
            type="text",
            text=(
                f"**Answer** (Trust: {data['trust_score']}/100):\n\n"
                f"{data['answer']}\n\n"
                f"**Sources**:\n{sources_str}\n\n"
                f"**Trust Breakdown**: {data.get('trust_breakdown', {})}"
            ),
        )
    ]


async def _upload(file_path: str, metadata: dict | None = None) -> list[TextContent]:
    result = await client.upload_document(file_path, metadata or {})
    return [
        TextContent(
            type="text",
            text=f"Uploaded: document_id={result['id']}, {result['num_chunks']} chunks indexed",
        )
    ]


async def _audit(
    limit: int = 10,
    max_trust_score: int | None = None,
    since_hours: int | None = None,
) -> list[TextContent]:
    entries = await client.get_audit_log(
        limit=limit, max_trust_score=max_trust_score, since_hours=since_hours
    )
    if not entries:
        return [TextContent(type="text", text="No audit entries matching criteria.")]
    formatted = "\n\n".join(
        f"**[{e['created_at']}] Trust: {e['trust_score']}**\n"
        f"Q: {e['question']}\n"
        f"A: {e['answer'][:200]}..."
        for e in entries
    )
    return [TextContent(type="text", text=formatted)]
