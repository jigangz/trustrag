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


REPHRASE_PROMPT = """Rephrase the following question in 3 different ways.
Keep the same meaning but change the wording significantly.
Return only the 3 rephrased questions, one per line.

Question: {question}
"""


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
        rag_fn: Function to run a question through the RAG pipeline

    Returns:
        {
            consistent: bool,
            score: float,  # 0-1, average pairwise similarity
            variants: [{question, answer, similarity_to_original}]
        }
    """
    # TODO: Generate rephrasings
    # TODO: Run each through RAG
    # TODO: Compare answers via embedding similarity
    # TODO: Flag if any pair has similarity < 0.85
    pass
