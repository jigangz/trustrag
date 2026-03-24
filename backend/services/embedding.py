"""Local embedding service using fastembed (no API key needed)."""

from fastembed import TextEmbedding

_model = TextEmbedding("BAAI/bge-small-en-v1.5")


async def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for a single text string."""
    return list(_model.embed([text]))[0].tolist()


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts."""
    return [e.tolist() for e in _model.embed(texts)]
