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


def _trust_score(data: dict) -> float:
    """Read trust score from either nested confidence.score or legacy trust_score key."""
    if "confidence" in data and isinstance(data["confidence"], dict):
        return data["confidence"].get("score", 0)
    return data.get("trust_score", 0)


def _trust_breakdown(data: dict) -> dict:
    if "confidence" in data and isinstance(data["confidence"], dict):
        return data["confidence"].get("breakdown", {})
    return data.get("trust_breakdown", {})


async def _query(
    question: str, min_trust_score: int = 0, top_k: int = 5
) -> list[TextContent]:
    data = await client.query(question, top_k=top_k)
    score = _trust_score(data)
    if score < min_trust_score:
        return [
            TextContent(
                type="text",
                text=(
                    f"No trustworthy answer available. "
                    f"Best match had trust score {score}, "
                    f"below threshold {min_trust_score}."
                ),
            )
        ]
    sources_str = "\n".join(
        # Backend SourceResponse uses 'document'; older shape used 'doc'. Support both.
        f"- {s.get('document') or s.get('doc', '?')} "
        f"(page {s.get('page', '?')}, similarity {s.get('similarity', 0):.2f})"
        for s in data["sources"][:5]
    )
    return [
        TextContent(
            type="text",
            text=(
                f"**Answer** (Trust: {score}/100):\n\n"
                f"{data['answer']}\n\n"
                f"**Sources**:\n{sources_str}\n\n"
                f"**Trust Breakdown**: {_trust_breakdown(data)}"
            ),
        )
    ]


async def _upload(file_path: str, metadata: dict | None = None) -> list[TextContent]:
    result = await client.upload_document(file_path, metadata or {})
    # Backend DocumentResponse uses 'total_chunks'; legacy shape used 'num_chunks'.
    chunks = result.get("total_chunks", result.get("num_chunks", 0))
    return [
        TextContent(
            type="text",
            text=f"Uploaded: document_id={result['id']}, {chunks} chunks indexed",
        )
    ]


def _entry_score(e: dict) -> float:
    return e.get("confidence_score", e.get("trust_score", 0)) or 0


def _entry_question(e: dict) -> str:
    return e.get("query") or e.get("question") or ""


async def _audit(
    limit: int = 10,
    max_trust_score: int | None = None,
    since_hours: int | None = None,
) -> list[TextContent]:
    # Backend only supports `limit` server-side. Fetch a wider window so client-side
    # filters still return up to `limit` matches.
    fetch_limit = max(limit * 3, 30) if (
        max_trust_score is not None or since_hours is not None
    ) else limit
    entries = await client.get_audit_log(limit=fetch_limit)

    # Client-side filter: trust threshold
    if max_trust_score is not None:
        entries = [e for e in entries if _entry_score(e) < max_trust_score]

    # Client-side filter: time window
    if since_hours is not None:
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        kept = []
        for e in entries:
            ts = e.get("created_at")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt >= cutoff:
                    kept.append(e)
            except (ValueError, TypeError):
                continue
        entries = kept

    entries = entries[:limit]
    if not entries:
        return [TextContent(type="text", text="No audit entries matching criteria.")]
    formatted = "\n\n".join(
        f"**[{e.get('created_at') or 'unknown time'}] Trust: {_entry_score(e)}**\n"
        f"Q: {_entry_question(e)}\n"
        f"A: {(e.get('answer') or '')[:200]}..."
        for e in entries
    )
    return [TextContent(type="text", text=formatted)]
