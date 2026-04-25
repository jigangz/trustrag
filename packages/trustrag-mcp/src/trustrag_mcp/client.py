"""HTTP client for communicating with the TrustRAG backend API."""

import httpx
from pathlib import Path


class TrustRAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    async def query(self, question: str, top_k: int = 5) -> dict:
        # Trailing slashes + follow_redirects handle FastAPI's APPEND_SLASH
        # behavior (307 /api/query → /api/query/). Without follow_redirects,
        # MCP clients choke on the 307 and report an opaque error.
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
            resp = await c.post(
                f"{self.base_url}/api/query/",
                json={"question": question, "top_k": top_k},
            )
            resp.raise_for_status()
            return resp.json()

    async def upload_document(self, file_path: str, metadata: dict) -> dict:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
            with open(file_path, "rb") as f:
                resp = await c.post(
                    f"{self.base_url}/api/documents/upload",
                    files={"file": (Path(file_path).name, f, "application/pdf")},
                    data={"metadata": str(metadata)},
                )
            resp.raise_for_status()
            return resp.json()

    async def get_audit_log(self, limit: int = 10) -> list[dict]:
        # Backend /api/audit/ only accepts (limit, offset) — see OpenAPI spec.
        # max_trust_score / since_hours filtering is applied client-side in server.py.
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            resp = await c.get(f"{self.base_url}/api/audit/", params={"limit": limit})
            resp.raise_for_status()
            return resp.json()
