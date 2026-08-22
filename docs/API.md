# Roshan RAG REST API

Base URL:

```text
http://localhost:8000/api/
```

The API is implemented with Django REST Framework. JSON requests use UTF-8;
document uploads use `multipart/form-data`. The browsable API is available at
the same endpoint URLs when the application is running.

The complete machine-readable contract is [openapi.yaml](openapi.yaml).

## Authentication and security

The demonstration configuration sets `AllowAny`, so these endpoints do not
require authentication. This is appropriate only for a local evaluation.
Configure authentication, authorization, HTTPS, and rate limiting before
exposing the API to other users or networks.

## Endpoint summary

| Method | Path | Success | Description |
| --- | --- | --- | --- |
| `GET` | `/api/documents/` | `200 OK` | List all documents. |
| `POST` | `/api/documents/` | `201 Created` | Upload a document. |
| `GET` | `/api/documents/{id}/` | `200 OK` | Retrieve one document. |
| `PUT` | `/api/documents/{id}/` | `200 OK` | Replace a document. |
| `PATCH` | `/api/documents/{id}/` | `200 OK` | Partially update a document. |
| `DELETE` | `/api/documents/{id}/` | `204 No Content` | Delete a document and its indexed chunks. |
| `POST` | `/api/ask/` | `201 Created` | Generate and store a grounded answer. |
| `GET` | `/api/history/` | `200 OK` | List stored questions and answers. |
| `GET` | `/api/history/{id}/` | `200 OK` | Retrieve a saved question and its sources. |

The current configuration returns unpaginated JSON arrays for document and
history lists. Both lists are ordered by creation time, newest first.

## Document representation

```json
{
  "id": 2,
  "title": "راهنمای استفاده از ownCloud",
  "file": "http://localhost:8000/media/documents/owncloud_user_guide_fa.docx",
  "text": "# راهنمای جامع سامانه ownCloud\n\n## ورود و امنیت حساب\n\n...",
  "chunk_count": 4,
  "created_at": "2026-08-22T09:15:00+03:30",
  "updated_at": "2026-08-22T09:15:00+03:30"
}
```

| Field | Type | Writable | Description |
| --- | --- | --- | --- |
| `id` | integer | No | Database identifier. |
| `title` | string | Yes | Display title, up to 255 characters. |
| `file` | URL or uploaded file | Yes | DOCX, TXT, or PDF file. |
| `text` | string | No | Extracted text including Word headings and tables. |
| `chunk_count` | integer | No | Number of indexed document chunks. |
| `created_at` | datetime | No | Creation timestamp. |
| `updated_at` | datetime | No | Most recent document update timestamp. |

### List documents

```http
GET /api/documents/
```

```bash
curl http://localhost:8000/api/documents/
```

Success: `200 OK` and an array of document objects.

### Upload a document

```http
POST /api/documents/
Content-Type: multipart/form-data
```

Required multipart fields:

- `title`: the document display name.
- `file`: a `.docx`, `.txt`, or `.pdf` file.

```bash
curl -X POST http://localhost:8000/api/documents/ \
  -F 'title=راهنمای استفاده از ownCloud' \
  -F 'file=@sample_data/owncloud_user_guide_fa.docx'
```

Success: `201 Created` and the saved document object. Uploading extracts the
file contents and schedules document indexing after the database transaction.

An unsupported file extension returns `400 Bad Request`:

```json
{
  "file": [
    "Only .docx, .txt and .pdf files are supported."
  ]
}
```

A corrupt file, an empty document, or a scanned PDF with no extractable text
also returns `400 Bad Request`. Run OCR on image-only PDFs before uploading.

### Retrieve one document

```http
GET /api/documents/2/
```

```bash
curl http://localhost:8000/api/documents/2/
```

Success: `200 OK` and the document object, including its extracted text.

### Partially update a document

Rename a document:

```bash
curl -X PATCH http://localhost:8000/api/documents/2/ \
  -H 'Content-Type: application/json' \
  -d '{"title":"راهنمای به‌روزشده ownCloud"}'
```

Replace the uploaded file:

```bash
curl -X PATCH http://localhost:8000/api/documents/2/ \
  -F 'file=@sample_data/support_and_pricing_fa.docx'
```

Success: `200 OK` and the updated document. Replacing the file extracts and
reindexes its new contents.

### Replace a document

```bash
curl -X PUT http://localhost:8000/api/documents/2/ \
  -F 'title=شرایط پشتیبانی' \
  -F 'file=@sample_data/support_and_pricing_fa.docx'
```

Success: `200 OK` and the replacement document.

### Delete a document

```bash
curl -X DELETE http://localhost:8000/api/documents/2/
```

Success: `204 No Content`. The uploaded file and its stored vector chunks are
removed. Previous questions remain, but their nullable `document` relation is
cleared.

## Ask a question

```http
POST /api/ask/
Content-Type: application/json; charset=utf-8
```

### Request fields

| Field | Type | Required | Rules |
| --- | --- | --- | --- |
| `question` | string | Yes | Non-empty question text. |
| `document_id` | positive integer or null | No | Search one document; omit or use `null` to search all documents. |
| `top_k` | integer | No | Number of returned chunks, from 1 to 10; defaults to 4. |

Search all indexed documents:

```bash
curl -X POST http://localhost:8000/api/ask/ \
  -H 'Content-Type: application/json; charset=utf-8' \
  -d '{
    "question": "فاکتور INV-2026-0456 چه مبلغی دارد؟",
    "top_k": 4
  }'
```

Restrict retrieval to one document:

```bash
curl -X POST http://localhost:8000/api/ask/ \
  -H 'Content-Type: application/json; charset=utf-8' \
  -d '{
    "question": "زمان پاسخ اولیه پشتیبانی Enterprise چقدر است؟",
    "document_id": 3,
    "top_k": 2
  }'
```

### Successful response

Status: `201 Created`.

```json
{
  "id": 8,
  "document": 3,
  "question": "زمان پاسخ اولیه پشتیبانی Enterprise چقدر است؟",
  "answer": "زمان پاسخ اولیه پشتیبانی Enterprise برابر با ۳۰ دقیقه است (بخش ۱).",
  "sources": [
    {
      "chunk_index": 1,
      "source": "document:3",
      "content": "سطح: Enterprise | پاسخ اولیه: ۳۰ دقیقه | شناسه SLA: SLA-ENT-247",
      "citation": "document:3 - شرایط پشتیبانی > سطوح پشتیبانی و SLA - chunk 1",
      "section": "شرایط پشتیبانی > سطوح پشتیبانی و SLA"
    }
  ],
  "created_at": "2026-08-22T09:30:00+03:30"
}
```

The `section` field is present only when the source chunk has heading metadata.
Source order matches the final reranked retrieval order. A globally scoped
question returns `"document": null`.

### Question and source fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | integer | Saved question identifier. |
| `document` | integer or null | Selected document; null means global search. |
| `question` | string | Original, unchanged user question. |
| `answer` | string | Generated Persian answer. |
| `sources` | array | Final reranked supporting chunks. |
| `sources[].chunk_index` | integer | Chunk position within its source document. |
| `sources[].source` | string | Source identifier such as `document:3`. |
| `sources[].content` | string | Original chunk text. |
| `sources[].citation` | string | Human-readable source citation. |
| `sources[].section` | string, optional | Hierarchical document heading path. |
| `created_at` | datetime | Question creation timestamp. |

### Validation errors

Missing question:

```json
{
  "question": [
    "This field is required."
  ]
}
```

Invalid `top_k`:

```json
{
  "top_k": [
    "Ensure this value is less than or equal to 10."
  ]
}
```

Unknown document:

```json
{
  "document_id": "Document not found."
}
```

Each of these responses uses `400 Bad Request`.

If retrieval, reranking configuration, or OpenRouter is unavailable, the ask
endpoint returns `503 Service Unavailable`:

```json
{
  "detail": "OPENROUTER_API_KEY is missing or still uses a placeholder."
}
```

The question is saved to history only after retrieval and answer generation
succeed, so a failed provider request does not leave an empty history record.

## Question history

### List previous questions

```bash
curl http://localhost:8000/api/history/
```

Success: `200 OK` and an array of saved question objects, newest first.

### Retrieve one saved answer

```bash
curl http://localhost:8000/api/history/8/
```

Success: `200 OK` and one complete question object including its sources.

History endpoints are read-only. Creating an answer is done through
`POST /api/ask/` or the Django Admin question form.

## Error status codes

| Status | Meaning |
| --- | --- |
| `400 Bad Request` | Missing or invalid fields, unsupported file, or unknown `document_id`. |
| `404 Not Found` | A document or history resource does not exist. |
| `405 Method Not Allowed` | The endpoint does not implement the requested HTTP method. |
| `503 Service Unavailable` | Retrieval, reranking configuration, or OpenRouter is unavailable. |
| `500 Internal Server Error` | An unhandled database, storage, or indexing failure. |

For API errors, inspect `docker compose logs -f web` and verify the database,
OpenRouter configuration, document index, and reranking model settings.

## Retrieval evaluation

Retrieval evaluation is an offline management command rather than a public API
endpoint. It never calls OpenRouter. Run the bundled BM25 baseline with:

```bash
python manage.py evaluate_retrieval --mode bm25 --top-k 4
```

Use `--mode hybrid` to evaluate the live PostgreSQL semantic index, weighted
RRF, BM25, and configured Cross-Encoder. The report includes Hit Rate, MRR,
mean Recall, mean Precision, per-query ranks, and a separate negative rejection
rate for `insufficient_context` examples.
