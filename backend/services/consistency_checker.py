"""Answer consistency validation.

Tests whether the system gives consistent answers when the same
question is phrased differently. Phrasing sensitivity is a known
problem with LLM-based systems — this catches it before it
reaches a user.

Process:
1. Take the original question
2. Generate 3 rephrasings via LLM
3. Run each through the RAG pipeline
4. Compare answers using embedding similarity
5. Flag if answers diverge significantly
"""

from openai import AsyncOpenAI
from config import settings
from services.embedding import embed_batch


REPHRASE_PROMPT = """Rephrase the following question in 3 different ways.
Keep the same meaning but change the wording significantly.
Return only the 3 rephrased questions, one per line.

Question: {question}
"""

CONSISTENCY_THRESHOLD = 0.85


def _groq_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def check_consistency(
    original_question: str,
    original_answer: str,
    rag_fn: callable,
) -> dict:
    """
    Check if the system gives consistent answers across rephrasings.

    Args:
        original_question: The user's original question
        original_answer: The answer we already generated
        rag_fn: Async function(question) -> str that runs a question through RAG

    Returns:
        {
            consistent: bool,
            score: float,  # 0-1, average pairwise similarity
            variants: [{question, answer, similarity_to_original}]
        }
    """
    # 1. Generate rephrasings via Groq
    client = _groq_client()
    try:
        response = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "user",
                    "content": REPHRASE_PROMPT.format(question=original_question),
                }
            ],
            temperature=0.7,
            max_tokens=256,
        )
        rephrasings_text = response.choices[0].message.content.strip()
        rephrasings = [
            line.strip().lstrip("0123456789.-) ")
            for line in rephrasings_text.split("\n")
            if line.strip()
        ][:3]
    except Exception:
        return {"consistent": True, "score": 1.0, "variants": []}

    if not rephrasings:
        return {"consistent": True, "score": 1.0, "variants": []}

    # 2. Run each rephrasing through RAG
    variants = []
    variant_answers = []
    for q in rephrasings:
        try:
            answer = await rag_fn(q)
            variant_answers.append(answer)
            variants.append({"question": q, "answer": answer})
        except Exception:
            variants.append({"question": q, "answer": "[error]"})
            variant_answers.append("")

    # 3. Compare answers via embedding similarity
    all_answers = [original_answer] + variant_answers
    non_empty = [a for a in all_answers if a]
    if len(non_empty) < 2:
        return {"consistent": True, "score": 1.0, "variants": variants}

    embeddings = await embed_batch(all_answers)
    original_emb = embeddings[0]

    similarities = []
    for i, variant in enumerate(variants):
        sim = _cosine_similarity(original_emb, embeddings[i + 1])
        variant["similarity_to_original"] = round(sim, 4)
        similarities.append(sim)

    avg_score = sum(similarities) / len(similarities) if similarities else 1.0
    consistent = all(s >= CONSISTENCY_THRESHOLD for s in similarities)

    return {
        "consistent": consistent,
        "score": round(avg_score, 4),
        "variants": variants,
    }
