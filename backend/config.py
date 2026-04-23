"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://trustrag:trustrag@db:5432/trustrag"

    # Groq (free tier — powers both LLM generation and trust verification)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_model: str = "llama-3.3-70b-versatile"

    # Local embeddings (fastembed, no API key needed)
    embedding_dimension: int = 384

    # Chunking
    chunk_size: int = 500  # tokens per chunk
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 5

    # Query cache
    query_cache_enabled: bool = False  # Feature flag for Postgres query cache
    query_cache_ttl_hours: int = 24

    # Merged generation + self-check prompt (Fix 3, HTTP only)
    merge_prompt_enabled: bool = False

    # Hybrid search
    hybrid_enabled: bool = True
    semantic_candidates: int = 20
    keyword_candidates: int = 20
    rrf_k: int = 60
    final_top_k: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
