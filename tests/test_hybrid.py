import unittest

from langchain_core.documents import Document

from rag.hybrid import HybridRetriever


class FakeVectorStore:
    def __init__(
        self,
        documents: list[Document],
    ) -> None:
        self.documents = documents

        self.last_search_kwargs = {}

    def similarity_search(
        self,
        query: str,
        **kwargs,
    ) -> list[Document]:
        self.last_search_kwargs = kwargs

        documents = self.documents

        source_filter = (
            kwargs
            .get(
                "filter",
                {},
            )
            .get(
                "source"
            )
        )

        if source_filter:
            documents = [
                document
                for document in documents
                if document.metadata.get(
                    "source"
                ) == source_filter
            ]

        return documents[
            : kwargs["k"]
        ]


class HybridRetrieverTests(
    unittest.TestCase
):
    @staticmethod
    def make_document(
        text: str,
        source: str,
        chunk_index: int = 0,
    ) -> Document:
        return Document(
            page_content=text,
            metadata={
                "source": source,
                "chunk_index": chunk_index,
            },
        )

    def test_bm25_finds_document_outside_vector_results(
        self,
    ) -> None:
        vector_document = self.make_document(
            "General cloud storage information.",
            "document:1",
        )

        exact_document = self.make_document(
            "The invoice reference is ZX-9182.",
            "document:2",
        )

        unrelated_document = self.make_document(
            "A separate guide about project configuration.",
            "document:3",
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                [
                    vector_document,
                ]
            ),
            lexical_documents=[
                vector_document,
                exact_document,
                unrelated_document,
            ],
            final_k=3,
        )

        results = retriever.invoke(
            "ZX-9182"
        )

        returned_sources = [
            document.metadata[
                "source"
            ]
            for document in results
        ]

        self.assertIn(
            "document:2",
            returned_sources,
        )

    def test_source_filter_applies_to_both_retrieval_paths(
        self,
    ) -> None:
        allowed_document = self.make_document(
            "The requested reference is ZX-9182.",
            "document:2",
        )

        blocked_document = self.make_document(
            "ZX-9182 belongs to another document.",
            "document:3",
        )

        vector_store = FakeVectorStore(
            [
                allowed_document,
                blocked_document,
            ]
        )

        retriever = HybridRetriever(
            vector_store,
            lexical_documents=[
                allowed_document,
                blocked_document,
            ],
            source="document:2",
            final_k=5,
        )

        results = retriever.invoke(
            "ZX-9182"
        )

        self.assertTrue(
            results
        )

        self.assertTrue(
            all(
                document.metadata[
                    "source"
                ] == "document:2"
                for document in results
            )
        )

        self.assertEqual(
            vector_store.last_search_kwargs[
                "filter"
            ],
            {
                "source": "document:2",
            },
        )

    def test_single_document_bm25_corpus_still_matches(
        self,
    ) -> None:
        exact_document = self.make_document(
            "Unique contract reference ZX-9182.",
            "document:4",
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                []
            ),
            lexical_documents=[
                exact_document,
            ],
            final_k=1,
        )

        results = retriever.invoke(
            "ZX-9182"
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].metadata[
                "source"
            ],
            "document:4",
        )

    def test_final_k_is_respected(
        self,
    ) -> None:
        documents = [
            self.make_document(
                f"Cloud storage document {index}.",
                f"document:{index}",
            )
            for index in range(
                5
            )
        ]

        retriever = HybridRetriever(
            FakeVectorStore(
                documents
            ),
            lexical_documents=documents,
            final_k=2,
        )

        results = retriever.invoke(
            "cloud"
        )

        self.assertEqual(
            len(results),
            2,
        )

    def test_empty_lexical_corpus_keeps_vector_results(
        self,
    ) -> None:
        vector_document = self.make_document(
            "Semantic search remains available.",
            "document:1",
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                [
                    vector_document,
                ]
            ),
            lexical_documents=[],
            final_k=1,
        )

        results = retriever.invoke(
            "semantic search"
        )

        self.assertEqual(
            results,
            [
                vector_document,
            ],
        )