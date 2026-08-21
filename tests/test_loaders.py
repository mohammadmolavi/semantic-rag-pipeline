import tempfile
import unittest
from pathlib import Path

from rag.langchain_rag import (
    chunk_ids_for_source,
    dedupe_documents,
    langchain_database_url,
)
from rag.llm import clean_llm_answer
from rag.loaders import load_document


class LoaderTests(unittest.TestCase):
    def test_chunk_ids_for_source(self) -> None:
        self.assertEqual(
            chunk_ids_for_source("document:3", 2),
            ["document:3:0", "document:3:1"],
        )

    def test_langchain_database_url_adds_psycopg(self) -> None:
        self.assertEqual(
            langchain_database_url(
                "postgresql://postgres:postgres@localhost:5432/roshan_rag"
            ),
            "postgresql+psycopg://postgres:postgres@localhost:5432/roshan_rag",
        )

    def test_load_document_reads_txt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "note.txt"
            path.write_text("hello rag", encoding="utf-8")
            self.assertEqual(load_document(path), "hello rag")

    def test_clean_llm_answer_drops_thinking(self) -> None:
        raw = (
            "Here's a thinking process:\n"
            "1. Analyze User Input\n"
            "NeRF is a model.\n"
            "بر اساس بخش ۰، نرف یک روش گرافیکی سه‌بعدی است."
        )
        cleaned = clean_llm_answer(raw)
        self.assertIn("نرف", cleaned)
        self.assertNotIn("thinking process", cleaned.lower())

    def test_dedupe_documents_keeps_first_copy(self) -> None:
        from langchain_core.documents import Document

        documents = [
            Document(page_content="same text", metadata={"source": "./txt.txt"}),
            Document(page_content="same text", metadata={"source": "document:1"}),
            Document(page_content="other", metadata={"source": "document:1"}),
        ]
        unique = dedupe_documents(documents)
        self.assertEqual(len(unique), 2)
        self.assertEqual(unique[0].metadata["source"], "./txt.txt")

