"""Trust verification layer — the core differentiator.

Computes a confidence score for every answer based on:
1. Retrieval similarity (40%) — How well do retrieved chunks match the query?
2. Source count (20%) — How many independent sources support the answer?
3. Source agreement (20%) — Do multiple sources say the same thing?
4. Hallucination check (20%) — Does the answer stay within source material?

Also runs a secondary LLM pass to detect hallucinated claims.
"""

from dataclasses import dataclass


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
Return a JSON array of flagged sentences with reasons.
If everything checks out, return an empty array.
"""


async def compute_trust_score(
    answer: str,
    sources: list[dict],
    query_embedding: list[float],
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
    # TODO: Implement scoring algorithm
    # 1. Average similarity of top sources (40% weight)
    # 2. Count distinct source documents (20% weight)
    # 3. Check if sources agree with each other (20% weight)
    # 4. Run hallucination detection LLM call (20% weight)
    pass


def _classify_confidence(score: float) -> str:
    """Map numeric score to confidence level."""
    if score >= 80:
        return "high"
    elif score >= 50:
        return "medium"
    else:
        return "low"
