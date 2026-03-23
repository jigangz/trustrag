"""Retrieval-Augmented Generation engine.

Orchestrates the full pipeline:
1. Embed user question
2. Retrieve relevant chunks
3. Build prompt with source context
4. Generate answer with source attribution
"""

from config import settings


SYSTEM_PROMPT = """You are a precise document assistant for construction safety.
Answer the question using ONLY the provided source documents.
For each claim in your answer, cite the source document and page number.

Rules:
- If the documents don't contain enough information, say so explicitly.
- Never make up information that isn't in the sources.
- If you're uncertain, indicate your uncertainty.
- Use the format [Source: document_name, p.XX] for citations.
"""


async def generate_answer(question: str, context_chunks: list[dict]) -> dict:
    """
    Generate an answer grounded in retrieved document chunks.

    Args:
        question: User's question
        context_chunks: Retrieved chunks with content, page, and document info

    Returns:
        {answer, sources_used, raw_response}
    """
    # TODO: Build prompt with context chunks
    # TODO: Call LLM (OpenAI/Anthropic)
    # TODO: Parse source citations from response
    pass
