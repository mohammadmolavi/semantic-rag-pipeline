# Roshan RAG

Minimal RAG prototype for document question answering.

## Setup

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
docker compose up -d postgres
```

Create `.env`:

```env
OPENROUTER_API_KEY=your_key
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/roshan_rag
```

## Run RAG Demo

The demo uses LangChain to read a `.txt` or `.docx` document, split it into
`Document` chunks, create SentenceTransformer embeddings, store them in PostgreSQL
with pgvector through `langchain-postgres`, retrieve the closest chunks, and ask
the LLM with that context.

```bash
python rag_demo.py ./sample.txt "سوال شما درباره سند چیست؟"
```
