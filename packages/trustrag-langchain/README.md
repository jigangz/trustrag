# trustrag-langchain

LangChain integration for TrustRAG — trust-verified retrieval with LangGraph multi-hop agent.

## Install

```bash
pip install trustrag-langchain
```

## Quick Start

### As Retriever

```python
from trustrag_langchain import TrustRAGRetriever

retriever = TrustRAGRetriever(
    endpoint="http://localhost:8000",
    min_trust_score=70,
)

docs = retriever.invoke("What's the fall protection height?")
for doc in docs:
    print(f"Trust: {doc.metadata['trust_score']}")
    print(doc.page_content)
```

### As Multi-Hop Agent

```python
from trustrag_langchain import TrustRAGRetriever, TrustBudgetAgent
from langchain_groq import ChatGroq

agent = TrustBudgetAgent(
    retriever=TrustRAGRetriever(endpoint="http://localhost:8000"),
    llm=ChatGroq(model="llama-3.3-70b-versatile"),
    min_trust_threshold=150,
    max_retrievals=3,
)

result = await agent.ainvoke("What PPE is required for work at heights above 6ft?")
print(f"Outcome: {result['outcome']}")  # "answer" or "stop_low_trust"
print(f"Cumulative trust: {result['cumulative_trust']}")
print(result["answer"])
```

## Why TrustRAG + LangChain?

- **Trust filtering**: Answers below threshold never enter your agent's context
- **Budget-aware agents**: Multi-hop reasoning stops early when evidence is insufficient
- **Citation chain**: Every claim traceable to source documents with trust scores
