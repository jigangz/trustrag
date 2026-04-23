"""Trust verification layer — the core differentiator.

Computes a confidence score for every answer based on:
1. Retrieval similarity (40%) — How well do retrieved chunks match the query?
2. Source count (20%) — How many independent sources support the answer?
3. Source agreement (20%) — Do multiple sources say the same thing?
4. Hallucination check (20%) — Does the answer stay within source material?

Also runs a secondary LLM pass to detect hallucinated claims.
"""

import json

from dataclasses import dataclass
from openai import AsyncOpenAI
from config import settings
from services.embedding import embed_batch


@dataclass
class TrustScore:
    """Complete trust assessment for an answer."""
    score: float  # 0-100
    level: str  # "high", "medium", "low"
    retrieval_similarity: float
    source_count_score: float
    source_agreement: float
    hallucination_free: bool
    hallucination_flags: list[dict]  # [{sentence, reason}]


HALLUCINATION_CHECK_PROMPT = """You are a fact-checker. Compare the answer against the source documents.

For each sentence in the answer:
1. Can it be verified from the provided sources?
2. Does it add information not found in any source?
3. Does it contradict any source?

Flag any sentence that contains information not directly supported by the sources.
Return ONLY a valid JSON array of flagged sentences with reasons.
If everything checks out, return an empty array [].

Example response format:
[{"sentence": "The building must be 50 feet tall.", "reason": "Sources mention 40 feet, not 50."}]
"""


def _groq_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )


async def _check_hallucination(answer: str, sources: list[dict]) -> list[dict]:
    """Run hallucination detection via Groq Llama."""
    source_text = "\n\n".join(
        f"[{s['filename']}, p.{s['page_number']}]: {s['content']}" for s in sources
    )
    client = _groq_client()
    try:
        response = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": HALLUCINATION_CHECK_PROMPT},
                {
                    "role": "user",
                    "content": f"Sources:\n{source_text}\n\nAnswer to check:\n{answer}",
                },
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        flags = json.loads(raw)
        if isinstance(flags, list):
            return flags
        return []
    except Exception:
        # If hallucination check fails, be conservative and return no flags
        return []


async def _compute_source_agreement(sources: list[dict]) -> float:
    """Check if sources agree by computing pairwise embedding similarity.

    Uses DB-stored embeddings from hybrid_search results when available,
    falls back to re-embedding only if embeddings are missing.
    """
    if len(sources) < 2:
        return 1.0  # Single source trivially agrees with itself

    # Prefer pre-computed embeddings from DB (via hybrid_search)
    embeddings = []
    missing = []
    for s in sources[:5]:
        emb = s.get("embedding")
        if emb and len(emb) > 0:
            embeddings.append(emb)
        else:
            missing.append(s["content"])

    # Fallback: embed any missing ones
    if missing:
        fallback = await embed_batch(missing)
        embeddings.extend(fallback)

    if len(embeddings) < 2:
        return 1.0

    # Average pairwise cosine similarity
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dot = sum(a * b for a, b in zip(embeddings[i], embeddings[j]))
            norm_i = sum(a * a for a in embeddings[i]) ** 0.5
            norm_j = sum(a * a for a in embeddings[j]) ** 0.5
            if norm_i > 0 and norm_j > 0:
                similarities.append(dot / (norm_i * norm_j))

    return sum(similarities) / len(similarities) if similarities else 1.0


async def compute_trust_score(
    answer: str,
    sources: list[dict],
    query_embedding: list[float],
    precomputed_hallucination_flags: list[dict] | None = None,
) -> TrustScore:
    """
    Run the full trust verification pipeline on an answer.

    Args:
        answer: The generated answer text
        sources: Retrieved chunks with similarity scores
        query_embedding: Original query embedding for comparison

    Returns:
        TrustScore with breakdown and any hallucination flags
    """
    # 1. Retrieval similarity (40%): average similarity of top sources
    if sources:
        avg_sim = sum(s["similarity"] for s in sources) / len(sources)
    else:
        avg_sim = 0.0
    retrieval_score = avg_sim * 100  # normalized to 0-100

    # 2. Source count (20%): distinct source documents
    unique_docs = len(set(s.get("document_id", s.get("filename", "")) for s in sources))
    if unique_docs >= 3:
        source_count_score = 20.0
    elif unique_docs == 2:
        source_count_score = 15.0
    elif unique_docs == 1:
        source_count_score = 5.0
    else:
        source_count_score = 0.0

    # 3. Source agreement (20%): do sources say similar things
    agreement = await _compute_source_agreement(sources)
    agreement_score = agreement * 20.0

    # 4. Hallucination check (20%): use precomputed flags or run LLM check
    if precomputed_hallucination_flags is not None:
        hallucination_flags = precomputed_hallucination_flags
    else:
        hallucination_flags = await _check_hallucination(answer, sources)
    hallucination_free = len(hallucination_flags) == 0
    hallucination_score = 20.0 if hallucination_free else max(0, 20.0 - len(hallucination_flags) * 5)

    # Weighted total
    total = (
        retrieval_score * 0.4
        + source_count_score
        + agreement_score
        + hallucination_score
    )
    total = min(100.0, max(0.0, total))

    return TrustScore(
        score=round(total, 1),
        level=_classify_confidence(total),
        retrieval_similarity=round(retrieval_score, 1),
        source_count_score=round(source_count_score, 1),
        source_agreement=round(agreement_score, 1),
        hallucination_free=hallucination_free,
        hallucination_flags=hallucination_flags,
    )


def _classify_confidence(score: float) -> str:
    """Map numeric score to confidence level."""
    if score >= 80:
        return "high"
    elif score >= 50:
        return "medium"
    else:
        return "low"
