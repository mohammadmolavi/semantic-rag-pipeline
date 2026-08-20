import tempfile
import unittest
from pathlib import Path

from rag.langchain_rag import chunk_ids_for_source, langchain_database_url
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
