"""Retrieval-Augmented Generation engine.

Orchestrates the full pipeline:
1. Embed user question
2. Retrieve relevant chunks
3. Build prompt with source context
4. Generate answer with source attribution
"""

import json
import logging
import re
from typing import AsyncIterator

from openai import AsyncOpenAI
from config import settings

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=settings.groq_api_key,
)

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


async def generate_answer_stream(
    question: str, context_chunks: list[dict]
) -> AsyncIterator[str]:
    """
    Stream answer tokens from Groq, yielding one token at a time.

    Uses the same prompt construction as generate_answer but with stream=True.
    Raises openai.RateLimitError on Groq 429.
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
        stream=True,
    )

    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


logger = logging.getLogger(__name__)

MERGED_SYSTEM_PROMPT = """You are a precise document assistant for construction safety.
Answer the question using ONLY the provided source documents, then perform a self-check
to identify any claims in your answer that are NOT directly supported by the sources.

Rules:
- Cite sources inline as [Source: document_name, p.XX]
- If sources lack sufficient information, say so explicitly
- Never fabricate page numbers or document names
- In self_check.unsupported_claims, list any sentence in your answer that cannot be
  fully verified from the sources provided

Return ONLY valid JSON matching this schema (no prose outside JSON):
{
  "answer": "the answer text with inline citations",
  "self_check": {
    "unsupported_claims": [
      {"sentence": "exact sentence text from answer", "reason": "why it's not supported"}
    ]
  }
}
"""


async def generate_answer_merged(question: str, context_chunks: list[dict]) -> dict:
    """Single-call generation + self-check via JSON structured output.

    Falls back to sequential generate_answer + _check_hallucination if JSON
    parse fails.

    Returns dict with keys: answer, sources_used, hallucination_flags,
    raw_response, merged (True if JSON parse succeeded, False if fallback used).
    """
    context = _build_context(context_chunks)

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": MERGED_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.2,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)

        answer = parsed["answer"]
        unsupported = parsed.get("self_check", {}).get("unsupported_claims", [])
        if not isinstance(unsupported, list):
            unsupported = []

        citations = _parse_citations(answer)
        return {
            "answer": answer,
            "sources_used": citations,
            "hallucination_flags": unsupported,
            "raw_response": response.model_dump(),
            "merged": True,
        }
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        logger.warning("Merged prompt JSON parse failed (%s), falling back to 2-call path", e)
        # Fallback: sequential generate + hallucination check
        from services.trust_verifier import _check_hallucination

        result = await generate_answer(question, context_chunks)
        flags = await _check_hallucination(result["answer"], context_chunks)
        return {
            "answer": result["answer"],
            "sources_used": result["sources_used"],
            "hallucination_flags": flags,
            "raw_response": result["raw_response"],
            "merged": False,
        }
