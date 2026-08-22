# Roshan RAG

Django document question-answering system. Upload `.docx` , `.txt` , `.pdf` files in
Admin or through the API, retrieve relevant chunks with LangChain + pgvector,
and answer questions with a free OpenRouter model.

## Setup

Create `.env`:

```env
OPENROUTER_API_KEY=your_key
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/roshan_rag
DJANGO_SUPERUSER_USERNAME=admin
DJANGO_SUPERUSER_PASSWORD=admin
DJANGO_SUPERUSER_EMAIL=admin@example.com
```

### Docker (recommended)

```bash
docker compose up --build
```

The first start downloads the embedding model and can take several minutes.

Then open:

- Admin: http://localhost:8000/admin/  (user `admin` / `admin`)
- API: http://localhost:8000/api/

### Local

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
docker compose up -d postgres
python manage.py migrate
python manage.py ensure_superuser
python manage.py load_sample_data
python manage.py runserver
```

## Using the admin

1. Sign in at `/admin/`.
2. Add a **Document** and upload a `.docx` file. The full text is stored and indexed.
3. Add a **Question**. Leave **Document** empty to search all files, or pick one file.
4. After save, the answer and retrieved chunks are stored in **Question** history.

## API

Base URL: `http://localhost:8000/api/`

Browsable API is enabled, so you can also open these URLs in a browser.

### Documents

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/documents/` | List documents |
| POST | `/api/documents/` | Upload a document (`multipart/form-data`: `title`, `file`) |
| GET | `/api/documents/{id}/` | Retrieve one document, including stored text |
| PATCH | `/api/documents/{id}/` | Update title or replace the file |
| DELETE | `/api/documents/{id}/` | Delete the document and its vectors |

Example:

```bash
curl -F "title=Sample" -F "file=@sample_data/neural_radiance_fields.docx" \
  http://localhost:8000/api/documents/
```

### Ask a question

`POST /api/ask/`

```json
{
  "question": "NeRF Ú†ÛŒØ³ØªØŸ",
  "document_id": 1,
  "top_k": 4
}
```

`document_id` is optional. Omit it to search every indexed document.
The request is stored in history.

### History

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/history/` | List questions and answers |
| GET | `/api/history/{id}/` | Retrieve one Q&A record |

## Sample data

`python manage.py load_sample_data` builds `sample_data/neural_radiance_fields.docx`
from `txt.txt` and uploads it if it is not already in the database.

## Testing retrieval and reranking

Run all unit and integration tests locally:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Run the same test suite inside the application container:

```bash
docker compose exec web python -m unittest discover -s tests -p 'test_*.py' -v
```

The integration suite uses the real `rank-bm25` implementation, LangChain's
`InMemoryVectorStore`, weighted reciprocal rank fusion, the production
`CrossEncoderReranker`, and actual DOCX extraction. Deterministic embeddings and
a deterministic scoring model keep the default run independent of PostgreSQL,
API credentials, network access, and model downloads.

To additionally download and run the actual configured Cross-Encoder model:

```bash
RUN_REAL_RERANKER_TESTS=1 RERANKER_DEVICE=cpu \
  python -m unittest tests.test_retrieval_integration.RealCrossEncoderModelTests -v
```

With Docker:

```bash
docker compose exec \
  -e RUN_REAL_RERANKER_TESTS=1 \
  -e RERANKER_DEVICE=cpu \
  web python -m unittest \
  tests.test_retrieval_integration.RealCrossEncoderModelTests -v
```

The optional model test requires `sentence-transformers` and may download the
model on its first execution. The default suite intentionally skips this test.

## CLI demo (optional)

```bash
python rag_demo.py ./txt.txt "Ø³ÙˆØ§Ù„ Ø´Ù…Ø§ Ø¯Ø±Ø¨Ø§Ø±Ù‡ Ø³Ù†Ø¯ Ú†ÛŒØ³ØªØŸ"
```