"""TrustRAG LangGraph agent with cumulative trust budget."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from trustrag_langchain.retriever import TrustRAGRetriever


class TrustBudgetState(TypedDict):
    question: str
    subqueries: list[str]
    retrievals: list[dict]
    cumulative_trust: float
    min_threshold: int
    max_retrievals: int
    final_answer: str | None
    outcome: Literal["answer", "stop_low_trust", "error"] | None


class TrustBudgetAgent:
    """Multi-hop LangGraph agent that tracks cumulative trust across retrievals.

    Three possible outcomes:
    - "answer": sufficient trust evidence gathered, synthesized answer returned
    - "stop_low_trust": max retrievals exhausted without meeting trust threshold
    - "error": unexpected failure during retrieval or synthesis
    """

    def __init__(
        self,
        retriever: TrustRAGRetriever,
        llm: BaseChatModel,
        min_trust_threshold: int = 150,
        max_retrievals: int = 3,
    ):
        self.retriever = retriever
        self.llm = llm
        self.min_threshold = min_trust_threshold
        self.max_retrievals = max_retrievals
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(TrustBudgetState)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("decide", self._decide_node)
        g.add_node("answer", self._answer_node)
        g.add_node("stop_low_trust", self._stop_low_trust_node)

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "decide")
        g.add_conditional_edges(
            "decide",
            self._route,
            {
                "retrieve": "retrieve",
                "answer": "answer",
                "stop_low_trust": "stop_low_trust",
            },
        )
        g.add_edge("answer", END)
        g.add_edge("stop_low_trust", END)
        return g.compile()

    async def _retrieve_node(self, state: TrustBudgetState) -> dict:
        next_q = state["subqueries"][-1] if state["subqueries"] else state["question"]
        try:
            docs = await self.retriever.ainvoke(next_q)
        except Exception:
            docs = []

        if docs:
            doc = docs[0]
            result = {
                "query": next_q,
                "answer": doc.page_content,
                "trust_score": doc.metadata["trust_score"],
                "sources": doc.metadata["sources"],
            }
        else:
            result = {"query": next_q, "answer": None, "trust_score": 0, "sources": []}

        return {
            "retrievals": [*state["retrievals"], result],
            "cumulative_trust": state["cumulative_trust"] + result["trust_score"],
        }

    async def _decide_node(self, state: TrustBudgetState) -> dict:
        # If cumulative trust already meets threshold, go straight to answer
        if state["cumulative_trust"] >= state["min_threshold"]:
            return {"final_answer": self._synthesize_answer(state)}

        # If max retrievals reached, routing will handle it
        if len(state["retrievals"]) >= state["max_retrievals"]:
            return {}

        # Ask LLM whether to answer or ask a sub-question
        prompt = self._build_decide_prompt(state)
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content.strip()

        if text.startswith("ANSWER"):
            return {"final_answer": self._synthesize_answer(state)}
        elif text.startswith("SUBQUERY:"):
            subquery = text.replace("SUBQUERY:", "").strip()
            return {"subqueries": [*state["subqueries"], subquery]}

        # Default: continue with original question
        return {}

    def _route(self, state: TrustBudgetState) -> str:
        # Already have a synthesized answer
        if state["final_answer"] is not None:
            return "answer"

        # Budget exhausted
        if len(state["retrievals"]) >= state["max_retrievals"]:
            if state["cumulative_trust"] >= state["min_threshold"]:
                return "answer"
            return "stop_low_trust"

        # Threshold met
        if state["cumulative_trust"] >= state["min_threshold"]:
            return "answer"

        # Continue retrieving
        return "retrieve"

    async def _answer_node(self, state: TrustBudgetState) -> dict:
        final = state["final_answer"] or self._synthesize_answer(state)
        return {"outcome": "answer", "final_answer": final}

    async def _stop_low_trust_node(self, state: TrustBudgetState) -> dict:
        msg = (
            f"Insufficient evidence to answer confidently "
            f"(cumulative trust: {state['cumulative_trust']:.0f}/{state['min_threshold']}). "
            f"Attempted {len(state['retrievals'])} retrievals. "
            f"Please consult a domain expert or rephrase your question."
        )
        return {"outcome": "stop_low_trust", "final_answer": msg}

    def _build_decide_prompt(self, state: TrustBudgetState) -> str:
        evidence = "\n\n".join(
            f"Query: {r['query']}\nAnswer: {r['answer']}\nTrust: {r['trust_score']}/100"
            for r in state["retrievals"]
        )
        return (
            f"Question: {state['question']}\n\n"
            f"Evidence gathered so far ({len(state['retrievals'])} retrievals, "
            f"cumulative trust: {state['cumulative_trust']:.0f}):\n\n{evidence}\n\n"
            f"Can you confidently answer the question with this evidence? "
            f"Reply exactly 'ANSWER' if yes. Otherwise reply 'SUBQUERY: <next sub-question>'."
        )

    def _synthesize_answer(self, state: TrustBudgetState) -> str:
        """Synthesize final answer, excluding retrievals with trust < 50."""
        parts = []
        for r in state["retrievals"]:
            if r["trust_score"] >= 50 and r["answer"]:
                sources_str = ", ".join(s["doc"] for s in r["sources"][:2])
                parts.append(
                    f"{r['answer']} (sources: {sources_str}, trust: {r['trust_score']})"
                )
        return "\n\n".join(parts) if parts else "No high-trust evidence found."

    async def ainvoke(self, question: str) -> dict:
        """Run the trust budget agent on a question.

        Returns:
            dict with keys: answer, outcome, cumulative_trust, retrievals
        """
        initial: TrustBudgetState = {
            "question": question,
            "subqueries": [],
            "retrievals": [],
            "cumulative_trust": 0.0,
            "min_threshold": self.min_threshold,
            "max_retrievals": self.max_retrievals,
            "final_answer": None,
            "outcome": None,
        }
        final = await self.graph.ainvoke(initial)
        return {
            "answer": final["final_answer"],
            "outcome": final["outcome"],
            "cumulative_trust": final["cumulative_trust"],
            "retrievals": final["retrievals"],
        }
