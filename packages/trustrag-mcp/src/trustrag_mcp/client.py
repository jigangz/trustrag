"""HTTP client for communicating with the TrustRAG backend API."""

import httpx
from pathlib import Path


class TrustRAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def query(self, question: str, top_k: int = 5) -> dict:
        async with httpx.AsyncClient(timeout=60) as c:
            resp = await c.post(
                f"{self.base_url}/api/query",
                json={"question": question, "top_k": top_k},
            )
            resp.raise_for_status()
            return resp.json()

    async def upload_document(self, file_path: str, metadata: dict) -> dict:
        async with httpx.AsyncClient(timeout=120) as c:
            with open(file_path, "rb") as f:
                resp = await c.post(
                    f"{self.base_url}/api/documents/upload",
                    files={"file": (Path(file_path).name, f, "application/pdf")},
                    data={"metadata": str(metadata)},
                )
            resp.raise_for_status()
            return resp.json()

    async def get_audit_log(
        self,
        limit: int = 10,
        max_trust_score: int | None = None,
        since_hours: int | None = None,
    ) -> list[dict]:
        params: dict = {"limit": limit}
        if max_trust_score is not None:
            params["max_trust_score"] = max_trust_score
        if since_hours is not None:
            params["since_hours"] = since_hours
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.get(f"{self.base_url}/api/audit", params=params)
            resp.raise_for_status()
            return resp.json()
