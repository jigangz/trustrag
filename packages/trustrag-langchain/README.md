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

### As Tool (for LangChain agents)

```python
from trustrag_langchain import TrustRAGTool

tool = TrustRAGTool()
result = tool.invoke({"query": "What PPE is required?"})
print(result)
```
