"""OpenAI embedding service for text vectorization."""

from openai import AsyncOpenAI
from config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key)


async def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for a single text string."""
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    return response.data[0].embedding


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts in a single API call."""
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]
