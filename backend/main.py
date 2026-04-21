"""TrustRAG — Construction AI you can trust.

FastAPI application entry point. Initializes the database,
registers routers, and configures CORS for the React frontend.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from routers import documents, query, audit, ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    await init_db()
    yield


app = FastAPI(
    title="TrustRAG",
    description="AI-powered document Q&A with built-in trust verification",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS — dev + production. Override via CORS_ORIGINS env (comma-separated).
_default_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://trustrag.vercel.app",
]
_env_origins = os.getenv("CORS_ORIGINS", "")
_allowed_origins = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins
    else _default_origins
)
# Also allow any *.vercel.app preview deployment via regex
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(audit.router, prefix="/api/audit", tags=["audit"])
app.include_router(ws.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "trustrag"}


# WS ping interval for uvicorn (used when running directly)
WS_PING_INTERVAL = 20


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=WS_PING_INTERVAL,
    )
