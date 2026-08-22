# Roshan RAG | Multilingual Document Question Answering

Roshan RAG is a Django-based retrieval-augmented generation application for
asking questions about uploaded **DOCX, TXT, and PDF documents**. It combines
PostgreSQL/pgvector semantic search, real BM25 keyword search, weighted
reciprocal rank fusion, and multilingual Cross-Encoder reranking. Answers retain
chunk-level citations and section names in both Django Admin and the REST API.

## Features

- Upload, replace, list, and delete documents through Django Admin or REST.
- Read Word headings, nested tables, merged cells, and paragraphs in their
  original document order.
- Split documents by heading and preserve hierarchical section metadata.
- Normalize Persian and Arabic characters, digits, diacritics, and half-spaces
  for lexical retrieval without altering the original document content.
- Combine semantic pgvector retrieval and independent BM25 keyword retrieval.
- Merge rankings with weighted reciprocal rank fusion and rerank candidate
  chunks with a multilingual Cross-Encoder.
- Answer in Persian using an OpenRouter-compatible chat model.
- Inspect source chunks, citations, and previous questions in Django Admin.
- Start PostgreSQL, pgvector, the application, sample data, and admin bootstrap
  with Docker Compose.

## Retrieval architecture

```mermaid
flowchart TD
    A[DOCX, TXT, or PDF upload] --> B[Ordered text and table extraction]
    B --> C[Section-aware chunking]
    C --> D[Semantic search with pgvector]
    C --> E[Lexical search with BM25]
    D --> F[Weighted reciprocal rank fusion]
    E --> F
    F --> G[Cross-Encoder reranking]
    G --> H[Grounded Persian answer and citations]
    H --> I[Django Admin and REST API]
```

For a candidate document `d`, fusion uses:

```text
RRF(d) = vector_weight / (rrf_k + vector_rank)
       + lexical_weight / (rrf_k + lexical_rank)
```

The default weights are `0.6` for semantic retrieval and `0.4` for BM25;
`rrf_k` defaults to `60`. A document appearing in both rankings receives both
contributions. Reranking runs before the final `top_k` cutoff.

## Project structure

```text
config/                         Django settings and URL configuration
documents/                      Models, Django Admin, API, and sample data
documents/management/commands/  Admin, sample-data, and retrieval-evaluation commands
docs/API.md                     Complete human-readable API reference
docs/openapi.yaml               OpenAPI 3.0 specification
docs/screenshots/               Screenshots captured from Django Admin
rag/                            Loaders, chunking, hybrid retrieval, evaluation, and LLM
sample_data/                    Bundled multilingual DOCX files and questions
tests/                          Unit, integration, and sample-data tests
docker-compose.yml              PostgreSQL/pgvector and Django services
```

## Requirements

- Docker and Docker Compose for the recommended installation.
- An OpenRouter API key and access to the configured model.
- For local development: Python 3.12 and a PostgreSQL instance with pgvector.
- Network access on the first run to install packages and download embedding
  and reranking models.

## Quick start with Docker

1. Create the local environment file:

   ```bash
   cp .env.example .env
   ```

2. Set `OPENROUTER_API_KEY` in `.env` and replace the development admin password.

3. Build and start the application:

   ```bash
   docker compose up --build
   ```

4. Wait for migrations, admin creation, sample-document loading, and model
   initialization to complete.

5. Open these URLs:

   - Django Admin: <http://localhost:8000/admin/>
   - Browsable API: <http://localhost:8000/api/>
   - Documents endpoint: <http://localhost:8000/api/documents/>
   - Question history: <http://localhost:8000/api/history/>

The sample configuration uses `admin` / `admin` only for local development.
Never expose those credentials or the development server publicly.

Useful container commands:

```bash
docker compose logs -f web
docker compose exec web python manage.py load_sample_data
docker compose exec web python manage.py evaluate_retrieval --mode hybrid
docker compose exec web python manage.py check
docker compose down
```

The PostgreSQL data, uploaded files, and Hugging Face model cache use named
Docker volumes and survive a normal container restart.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
docker compose up -d postgres
python manage.py migrate
python manage.py ensure_superuser
python manage.py load_sample_data
python manage.py runserver
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Configuration

| Variable | Example or default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | `replace-me` | API key used for answer generation. |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter model identifier. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible API base URL. |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/roshan_rag` | PostgreSQL connection string. |
| `DJANGO_SECRET_KEY` | Change for deployment | Django signing and cryptographic secret. |
| `DJANGO_DEBUG` | `true` | Development debug mode; disable in production. |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated permitted hosts. |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | Trusted browser origins. |
| `DJANGO_SUPERUSER_USERNAME` | `admin` | Bootstrap administrator username. |
| `DJANGO_SUPERUSER_PASSWORD` | `admin` | Bootstrap administrator password. |
| `DJANGO_SUPERUSER_EMAIL` | `admin@example.com` | Bootstrap administrator email. |
| `INDEX_DOCUMENTS` | `true` | Enable document vector indexing after upload. |
| `RERANKER_ENABLED` | `true` | Enable multilingual Cross-Encoder reranking. |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Reranking model identifier. |
| `RERANKER_DEVICE` | Empty for automatic selection | Optional device such as `cpu`. |
| `RERANKER_BATCH_SIZE` | `16` | Number of question/document pairs per inference batch. |

The Docker `web` service overrides `DATABASE_URL` to reach the `postgres`
container. Keep `.env` local; it is excluded by `.gitignore` and `.dockerignore`.

## Bundled sample documents

Run `python manage.py load_sample_data` to create and register the following
documents. The command is idempotent and does not depend on an untracked
`txt.txt` file.

| Document                     | File | Demonstrates |
|------------------------------| --- | --- |
| Neural Radiance Fields       | `sample_data/neural_radiance_fields.docx` | English technical content and semantic search. |
| راهنمای استفاده از ownCloud  | `sample_data/owncloud_user_guide_fa.docx` | Persian headings, pricing tables, MFA, and plan identifiers. |
| شرایط پشتیبانی و قیمت‌گذاری  | `sample_data/support_and_pricing_fa.docx` | SLA tables, invoices, exact identifiers, and support policies. |

`sample_data/sample_questions.json` contains reproducible example questions,
expected answer fragments, source files, and one intentionally unrelated
question for grounding checks.

Example questions:

```text
مبلغ ماهانه پلن حرفه‌ای Pro چقدر است؟
کد سرویس OC-PRO-500 مربوط به کدام پلن است؟
زمان پاسخ اولیه پشتیبانی Enterprise چقدر است؟
فاکتور INV-2026-0456 چه مبلغی دارد؟
نسخه پشتیبان اطلاعات چند روز نگهداری می‌شود؟
What rendering technique does NeRF use?
```

The pricing and support identifiers are fictional demonstration data, not
official ownCloud product prices or service commitments.

## Retrieval evaluation

Retrieval can be evaluated independently from answer generation, so neither an
OpenRouter request nor LLM-as-a-judge is needed. The evaluator reads
`sample_data/sample_questions.json`, derives relevant chunk IDs from each
expected answer fragment, and reports `Hit Rate@K`, `MRR@K`, mean `Recall@K`,
and mean `Precision@K`.

Run the deterministic BM25 baseline without PostgreSQL or model downloads:

```bash
python manage.py evaluate_retrieval --mode bm25 --top-k 4
```

Evaluate the live semantic + BM25 + RRF + Cross-Encoder pipeline after the
sample documents have been indexed:

```bash
python manage.py evaluate_retrieval --mode hybrid --top-k 4
```

Useful options:

```bash
# Compare hybrid retrieval without the Cross-Encoder.
python manage.py evaluate_retrieval --mode hybrid --no-reranker

# Produce JSON for CI and fail when Hit Rate@4 falls below 90%.
python manage.py evaluate_retrieval --mode bm25 --json --min-hit-rate 0.90
```

Positive queries contribute to ranking metrics. Queries labeled
`insufficient_context` are reported separately as the negative rejection rate.
That negative metric measures retrieval abstention; factuality and faithfulness
of the generated answer require a separate answer-level evaluation.

## Django Admin workflow

1. Sign in at `/admin/`.
2. Open **Documents** to inspect the bundled files or upload a `.docx`, `.txt`,
   or `.pdf` document.
3. Open an existing document to review extracted headings, table content, and
   its indexed chunk count.
4. Open **Questions**, create a question, and optionally select one document.
5. Leave **Document** empty to search across every indexed file.
6. Inspect the generated Persian answer, section path, chunk numbers, and
   source text on the saved question page.

### Admin dashboard

![Django Admin dashboard](docs/screenshots/admin-dashboard.png)

### Indexed sample documents

![Django Admin document list](docs/screenshots/admin-documents.png)

### Structured document detail

![Django Admin document detail](docs/screenshots/admin-document-detail.png)

### Persian answer and cited sources

![Django Admin question and answer](docs/screenshots/admin-question-answer.png)

## REST API

The API root is `http://localhost:8000/api/`. Complete request and response
examples, validation errors, field definitions, and status codes are documented
in [docs/API.md](docs/API.md). The machine-readable contract is available at
[docs/openapi.yaml](docs/openapi.yaml).

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/documents/` | List uploaded documents. |
| `POST` | `/api/documents/` | Upload a DOCX, TXT, or PDF document. |
| `GET` | `/api/documents/{id}/` | Retrieve one document and its extracted text. |
| `PUT` | `/api/documents/{id}/` | Replace a document. |
| `PATCH` | `/api/documents/{id}/` | Update document fields or replace its file. |
| `DELETE` | `/api/documents/{id}/` | Remove a document, uploaded file, and vectors. |
| `POST` | `/api/ask/` | Ask a question across one or all documents. |
| `GET` | `/api/history/` | List previous questions and generated answers. |
| `GET` | `/api/history/{id}/` | Retrieve one saved question and its citations. |

Upload a document:

```bash
curl -X POST http://localhost:8000/api/documents/ \
  -F 'title=راهنمای ownCloud' \
  -F 'file=@sample_data/owncloud_user_guide_fa.docx'
```

Ask a grounded question:

```bash
curl -X POST http://localhost:8000/api/ask/ \
  -H 'Content-Type: application/json' \
  -d '{"question":"مبلغ پلن حرفه‌ای Pro چقدر است؟","top_k":4}'
```

The demo API currently allows unauthenticated requests. Add appropriate
authentication, authorization, rate limiting, and HTTPS before deployment.

## Test suite

Run the entire suite locally:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Run it inside Docker:

```bash
docker compose exec web python -m unittest discover -s tests -p 'test_*.py' -v
```

The default tests use real `rank-bm25`, LangChain's `InMemoryVectorStore`,
production RRF and reranking code, actual DOCX generation, nested tables,
Persian normalization, retrieval metrics, sample data, API failure handling,
file replacement/deletion, and source serialization. PostgreSQL, OpenRouter
credentials, and model downloads are not required for the fast suite.

Run the optional actual Cross-Encoder model test:

```bash
RUN_REAL_RERANKER_TESTS=1 RERANKER_DEVICE=cpu \
  python -m unittest \
  tests.test_retrieval_integration.RealCrossEncoderModelTests -v
```

The first optional run can download the configured model.

## Troubleshooting

- **The first startup is slow:** embedding and reranking models are downloaded
  once and then reused from the Hugging Face Docker volume.
- **PostgreSQL connection fails:** confirm that `docker compose ps` reports a
  healthy `postgres` service and that `DATABASE_URL` matches the run mode.
- **Answers fail or remain empty:** check `OPENROUTER_API_KEY`, the configured
  model name, and `docker compose logs -f web`.
- **A document has zero chunks:** verify that `INDEX_DOCUMENTS=true` and that
  the document contains extractable text.
- **A scanned PDF upload is rejected:** the PDF loader extracts embedded text;
  run OCR first because image-only PDFs contain no searchable text.
- **Old documents do not include table content:** replace or re-upload their
  files after deploying the structured DOCX loader.
- **Reranking is too expensive:** set `RERANKER_DEVICE=cpu`, lower
  `RERANKER_BATCH_SIZE`, or set `RERANKER_ENABLED=false`.
- **Persian text appears corrupted:** keep repository files and API request
  bodies encoded as UTF-8; do not save Persian source files as Windows-1252.

## Deployment considerations

The included Compose file and `runserver` entrypoint target a demonstration
environment. For production, disable debug mode, rotate the Django secret and
admin password, protect the API, terminate TLS, restrict allowed hosts, and use
a production WSGI/ASGI server with managed PostgreSQL and backup policies.

## Command-line demo

```bash
python rag_demo.py sample_data/owncloud_user_guide_fa.docx \
  'مبلغ ماهانه پلن حرفه‌ای Pro چقدر است؟'
```
