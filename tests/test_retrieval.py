import unittest

from rag.retrieval import KeywordRetriever, split_text


class RetrievalTests(unittest.TestCase):
    def test_split_text_uses_overlap(self) -> None:
        chunks = split_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=3)
        self.assertEqual(chunks, ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"])

    def test_keyword_retriever_returns_relevant_chunk(self) -> None:
        retriever = KeywordRetriever.from_texts(
            [
                "Django admin is used for document management.",
                "PostgreSQL stores vectors with pgvector.",
                "OpenRouter connects the app to language models.",
            ],
            source="sample",
        )
        results = retriever.search("How do we store vectors?", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].index, 1)
