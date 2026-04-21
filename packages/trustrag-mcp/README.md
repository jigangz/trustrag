# trustrag-mcp

MCP server for TrustRAG — trust-verified RAG exposed as tools for Claude Desktop, Cursor, and other MCP clients.

## Installation

```bash
pip install trustrag-mcp
```

## Usage

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "trustrag": {
      "command": "uvx",
      "args": ["trustrag-mcp"],
      "env": {
        "TRUSTRAG_BACKEND_URL": "http://localhost:8000"
      }
    }
  }
}
```

### Direct

```bash
# Via entry point
trustrag-mcp

# Via module
python -m trustrag_mcp
```

## Tools

| Tool | Description |
|------|-------------|
| `trustrag_query` | Ask the knowledge base with trust filtering |
| `trustrag_upload_document` | Upload a PDF to the knowledge base |
| `trustrag_get_audit_log` | Fetch recent query audit entries |

## Configuration

Set `TRUSTRAG_BACKEND_URL` environment variable (default: `http://localhost:8000`).
