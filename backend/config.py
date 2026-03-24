"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://trustrag:trustrag@db:5432/trustrag"

    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    # Groq (for hallucination detection & consistency checks)
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-70b-versatile"

    # Chunking
    chunk_size: int = 500  # tokens per chunk
    chunk_overlap: int = 50

    # Retrieval
    top_k: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
