import unittest

from langchain_core.documents import Document

from rag.hybrid import HybridRetriever

from unittest.mock import Mock


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
    def test_bm25_matches_arabic_and_persian_character_variants(
        self,
    ) -> None:
        document = self.make_document(
            "راهنمای كتابخانه مرکزی",
            "document:10",
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                []
            ),
            lexical_documents=[
                document,
            ],
            final_k=1,
        )

        results = retriever.invoke(
            "کتابخانه"
        )

        self.assertEqual(
            results,
            [
                document,
            ],
        )

    def test_bm25_matches_persian_and_arabic_digits(
        self,
    ) -> None:
        document = self.make_document(
            "شماره قرارداد ٤٥٦",
            "document:11",
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                []
            ),
            lexical_documents=[
                document,
            ],
            final_k=1,
        )

        results = retriever.invoke(
            "۴۵۶"
        )

        self.assertEqual(
            results,
            [
                document,
            ],
        )

    def test_bm25_matches_half_space_and_regular_space(
        self,
    ) -> None:
        document = self.make_document(
            "سامانه به روز می\u200cشود",
            "document:12",
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                []
            ),
            lexical_documents=[
                document,
            ],
            final_k=1,
        )

        results = retriever.invoke(
            "می شود"
        )

        self.assertEqual(
            results,
            [
                document,
            ],
        )
    def test_rrf_prioritizes_documents_found_by_both_retrievers(
        self,
    ) -> None:
        vector_only = self.make_document(
            "Semantic-only result.",
            "document:20",
        )

        shared = self.make_document(
            "Result found by both retrieval methods.",
            "document:21",
        )

        lexical_only = self.make_document(
            "Keyword-only result.",
            "document:22",
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                []
            ),
        )

        ranked = retriever._reciprocal_rank_fusion(
            [
                vector_only,
                shared,
            ],
            [
                (
                    lexical_only,
                    100.0,
                ),
                (
                    shared,
                    1.0,
                ),
            ],
        )

        self.assertEqual(
            [
                document
                for document, _ in ranked
            ],
            [
                shared,
                vector_only,
                lexical_only,
            ],
        )

    def test_rrf_applies_expected_weighted_formula(
        self,
    ) -> None:
        vector_only = self.make_document(
            "Vector result.",
            "document:23",
        )

        shared = self.make_document(
            "Shared result.",
            "document:24",
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                []
            ),
            rrf_k=60,
            vector_weight=0.6,
            lexical_weight=0.4,
        )

        ranked = retriever._reciprocal_rank_fusion(
            [
                vector_only,
                shared,
            ],
            [
                (
                    shared,
                    9.0,
                ),
            ],
        )

        scores = {
            document.metadata[
                "source"
            ]: score
            for document, score in ranked
        }

        self.assertAlmostEqual(
            scores[
                "document:23"
            ],
            0.6 / 61,
        )

        self.assertAlmostEqual(
            scores[
                "document:24"
            ],
            (
                0.6 / 62
            )
            +
            (
                0.4 / 61
            ),
        )

    def test_rrf_merges_duplicate_chunks_across_retrieval_paths(
        self,
    ) -> None:
        vector_document = self.make_document(
            "Shared contract information.",
            "document:25",
            chunk_index=3,
        )

        lexical_document = self.make_document(
            "Shared contract information.",
            "document:25",
            chunk_index=3,
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                []
            ),
        )

        ranked = retriever._reciprocal_rank_fusion(
            [
                vector_document,
            ],
            [
                (
                    lexical_document,
                    5.0,
                ),
            ],
        )

        self.assertEqual(
            len(
                ranked
            ),
            1,
        )

        self.assertIs(
            ranked[0][0],
            vector_document,
        )

        self.assertAlmostEqual(
            ranked[0][1],
            1 / 61,
        )

    def test_rrf_ignores_duplicate_chunks_within_one_ranking(
        self,
    ) -> None:
        document = self.make_document(
            "Repeated semantic result.",
            "document:26",
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                []
            ),
        )

        ranked = retriever._reciprocal_rank_fusion(
            [
                document,
                document,
            ],
            [],
        )

        self.assertEqual(
            len(
                ranked
            ),
            1,
        )

        self.assertAlmostEqual(
            ranked[0][1],
            0.6 / 61,
        )

    def test_rrf_respects_custom_retrieval_weights(
        self,
    ) -> None:
        vector_document = self.make_document(
            "Semantic result.",
            "document:27",
        )

        lexical_document = self.make_document(
            "Keyword result.",
            "document:28",
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                []
            ),
            vector_weight=0.1,
            lexical_weight=0.9,
        )

        ranked = retriever._reciprocal_rank_fusion(
            [
                vector_document,
            ],
            [
                (
                    lexical_document,
                    0.1,
                ),
            ],
        )

        self.assertEqual(
            ranked[0][0],
            lexical_document,
        )

    def test_rrf_skips_retrieval_paths_with_zero_weight(
        self,
    ) -> None:
        vector_document = self.make_document(
            "Semantic result.",
            "document:29",
        )

        lexical_document = self.make_document(
            "Keyword result.",
            "document:30",
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                []
            ),
            vector_weight=0.0,
            lexical_weight=1.0,
        )

        ranked = retriever._reciprocal_rank_fusion(
            [
                vector_document,
            ],
            [
                (
                    lexical_document,
                    3.0,
                ),
            ],
        )

        self.assertEqual(
            [
                document
                for document, _ in ranked
            ],
            [
                lexical_document,
            ],
        )

    def test_rrf_rejects_invalid_fusion_settings(
        self,
    ) -> None:
        invalid_settings = [
            {
                "rrf_k": 0,
            },
            {
                "vector_weight": -0.1,
            },
            {
                "lexical_weight": -0.1,
            },
            {
                "vector_weight": 0.0,
                "lexical_weight": 0.0,
            },
        ]

        for settings in invalid_settings:
            with self.subTest(
                settings=settings
            ):
                with self.assertRaises(
                    ValueError
                ):
                    HybridRetriever(
                        FakeVectorStore(
                            []
                        ),
                        **settings,
                    )
    def test_reranker_receives_rrf_candidates_before_final_cutoff(
        self,
    ) -> None:
        documents = [
            self.make_document(
                f"Candidate {index}.",
                f"document:{40 + index}",
            )
            for index in range(
                3
            )
        ]

        reranker = Mock()

        reranker.rerank.return_value = list(
            reversed(
                documents
            )
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                documents
            ),
            lexical_documents=[],
            reranker=reranker,
            rerank_k=3,
            final_k=1,
        )

        results = retriever.invoke(
            "candidate"
        )

        reranker.rerank.assert_called_once_with(
            "candidate",
            documents,
        )

        self.assertEqual(
            results,
            [
                documents[2],
            ],
        )

    def test_reranker_candidate_limit_is_respected(
        self,
    ) -> None:
        documents = [
            self.make_document(
                f"Candidate {index}.",
                f"document:{50 + index}",
            )
            for index in range(
                4
            )
        ]

        reranker = Mock()

        reranker.rerank.side_effect = (
            lambda question, candidates: candidates
        )

        retriever = HybridRetriever(
            FakeVectorStore(
                documents
            ),
            lexical_documents=[],
            reranker=reranker,
            rerank_k=2,
            final_k=2,
        )

        results = retriever.invoke(
            "candidate"
        )

        reranker.rerank.assert_called_once_with(
            "candidate",
            documents[:2],
        )

        self.assertEqual(
            results,
            documents[:2],
        )

    def test_invalid_rerank_candidate_limit_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            HybridRetriever(
                FakeVectorStore(
                    []
                ),
                rerank_k=0,
            )