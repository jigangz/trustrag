# TrustRAG

> AI-powered document Q&A with built-in trust verification for high-stakes environments.

## Problem

Construction teams are told to adopt AI, but most tools hallucinate, cite wrong sources, and offer no audit trail. A foreman can't trust a system that confidently gives the wrong answer on a job site where mistakes have real consequences.

**TrustRAG doesn't just answer questions — it proves why you should trust the answer.**

## Completely Free to Run

TrustRAG requires **no paid API keys**:
- **LLM**: Groq (free tier) — Llama 3.1 70B for generation and trust verification
- **Embeddings**: fastembed (local) — BAAI/bge-small-en-v1.5 runs on CPU, no API key needed
- **Vector Store**: pgvector (self-hosted via Docker)

You only need a free [Groq API key](https://console.groq.com).

## Features

- **Document Ingestion** — Upload PDF manuals, specs, and safety documents. Automatically parsed, chunked, and embedded.
- **Retrieval-Augmented Generation** — Answers grounded in your actual documents, not the model's training data.
- **Confidence Scoring** — Every answer gets a 0-100 trust score based on retrieval quality, source agreement, and hallucination detection.
- **Source Tracing** — Every claim linked back to the exact document, page, and paragraph.
- **Hallucination Detection** — Secondary LLM pass flags any claims not supported by retrieved sources.
- **Answer Consistency Check** — Same question rephrased 3 ways; inconsistent answers get flagged.
- **Full Audit Trail** — Every query logged: who asked, what was retrieved, what the model said, and why it was trusted or flagged.

## Architecture

```
User uploads PDF → Parse & chunk → Embed locally (fastembed) → Store (pgvector)
                                                                    ↓
User asks question → Retrieve top-k chunks → LLM generates answer (Groq) → Trust verification → Response with audit
                                                                                  ↓
                                                                           Audit log (PostgreSQL)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) |
| LLM | Groq (Llama 3.1 70B) — free tier |
| Embeddings | fastembed (BAAI/bge-small-en-v1.5) — local, no API key |
| Vector Store | pgvector (PostgreSQL) |
| Frontend | React (Vite) + Tailwind CSS |
| Audit Storage | PostgreSQL |
| PDF Parsing | pdfplumber |
| Deployment | Docker Compose |

## Quick Start

```bash
# Clone
git clone https://github.com/jigangz/trustrag.git
cd trustrag

# Configure
cp .env.example .env
# Add your free Groq API key to .env

# Run
docker-compose up --build

# Open
# Frontend: http://localhost:5173
# API docs: http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/documents/upload` | Upload and process a PDF |
| GET | `/api/documents` | List uploaded documents |
| DELETE | `/api/documents/{id}` | Remove a document |
| POST | `/api/query` | Ask a question with trust verification |
| GET | `/api/audit` | View query audit trail |
| GET | `/api/audit/{id}` | Get detailed audit for a specific query |

## Trust Score Breakdown

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Retrieval Similarity | 40% | How closely retrieved chunks match the query |
| Source Count | 20% | Number of independent sources supporting the answer |
| Source Agreement | 20% | Whether multiple sources say the same thing |
| Hallucination Check | 20% | Whether the answer stays within source material |

### Confidence Levels

- **80-100** — High confidence. Answer verified against multiple sources.
- **50-79** — Medium confidence. Limited sources or partial coverage. Human review recommended.
- **0-49** — Low confidence. Insufficient evidence or potential hallucination detected. Do not use without manual verification.

## Project Structure

```
trustrag/
├── backend/
│   ├── main.py                    # FastAPI entry point
│   ├── config.py                  # Environment configuration
│   ├── models.py                  # SQLAlchemy + Pydantic models
│   ├── database.py                # Database connection + init
│   ├── routers/
│   │   ├── documents.py           # Document upload & management
│   │   ├── query.py               # Q&A with trust verification
│   │   └── audit.py               # Audit trail endpoints
│   ├── services/
│   │   ├── document_processor.py  # PDF parsing & chunking
│   │   ├── embedding.py           # Local embedding (fastembed)
│   │   ├── vector_store.py        # pgvector operations
│   │   ├── rag_engine.py          # Retrieval + LLM generation (Groq)
│   │   ├── trust_verifier.py      # Confidence scoring + hallucination detection
│   │   └── consistency_checker.py # Answer consistency validation
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Layout.jsx
│   │   │   ├── DocumentUpload.jsx
│   │   │   ├── DocumentList.jsx
│   │   │   ├── QueryPanel.jsx
│   │   │   ├── AnswerCard.jsx
│   │   │   ├── ConfidenceBadge.jsx
│   │   │   ├── SourceCard.jsx
│   │   │   ├── AuditTimeline.jsx
│   │   │   └── ConsistencyView.jsx
│   │   └── hooks/
│   │       ├── useDocuments.js
│   │       ├── useQuery.js
│   │       └── useAudit.js
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## License

MIT
