"""Retrieval-Augmented Generation engine.

Orchestrates the full pipeline:
1. Embed user question
2. Retrieve relevant chunks
3. Build prompt with source context
4. Generate answer with source attribution
"""

import re

from openai import AsyncOpenAI
from config import settings

client = AsyncOpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """You are a precise document assistant for construction safety.
Answer the question using ONLY the provided source documents.
For each claim in your answer, cite the source document and page number.

Rules:
- If the documents don't contain enough information, say so explicitly.
- Never make up information that isn't in the sources.
- If you're uncertain, indicate your uncertainty.
- Use the format [Source: document_name, p.XX] for citations.
"""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"--- Source {i}: {c['filename']}, page {c['page_number']} ---\n{c['content']}"
        )
    return "\n\n".join(parts)


def _parse_citations(text: str) -> list[dict]:
    pattern = r'\[Source:\s*([^,\]]+),\s*p\.?\s*(\d+)\]'
    citations = []
    seen = set()
    for match in re.finditer(pattern, text):
        doc = match.group(1).strip()
        page = int(match.group(2))
        key = (doc, page)
        if key not in seen:
            seen.add(key)
            citations.append({"document": doc, "page": page})
    return citations


async def generate_answer(question: str, context_chunks: list[dict]) -> dict:
    """
    Generate an answer grounded in retrieved document chunks.

    Args:
        question: User's question
        context_chunks: Retrieved chunks with content, page, and document info

    Returns:
        {answer, sources_used, raw_response}
    """
    context = _build_context(context_chunks)

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    citations = _parse_citations(answer)

    return {
        "answer": answer,
        "sources_used": citations,
        "raw_response": response.model_dump(),
    }
