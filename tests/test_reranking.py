import os
import unittest

from unittest.mock import Mock, patch

from langchain_core.documents import Document

from rag.reranking import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
)


class CrossEncoderRerankerTests(
    unittest.TestCase
):
    @staticmethod
    def make_document(
        text: str,
        source: str,
    ) -> Document:
        return Document(
            page_content=text,
            metadata={
                "source": source,
                "chunk_index": 0,
            },
        )

    @patch(
        "rag.reranking._load_cross_encoder"
    )
    def test_reranker_orders_documents_by_cross_encoder_score(
        self,
        load_model,
    ) -> None:
        first = self.make_document(
            "General information.",
            "document:1",
        )

        second = self.make_document(
            "Exact answer to the question.",
            "document:2",
        )

        third = self.make_document(
            "Related background.",
            "document:3",
        )

        model = Mock()

        model.predict.return_value = [
            0.2,
            0.9,
            0.5,
        ]

        load_model.return_value = model

        reranker = CrossEncoderReranker(
            batch_size=8
        )

        results = reranker.rerank(
            "What is the answer?",
            [
                first,
                second,
                third,
            ],
        )

        self.assertEqual(
            results,
            [
                second,
                third,
                first,
            ],
        )

        model.predict.assert_called_once_with(
            [
                (
                    "What is the answer?",
                    "General information.",
                ),
                (
                    "What is the answer?",
                    "Exact answer to the question.",
                ),
                (
                    "What is the answer?",
                    "Related background.",
                ),
            ],
            batch_size=8,
            show_progress_bar=False,
        )

    @patch(
        "rag.reranking._load_cross_encoder"
    )
    def test_empty_and_single_candidates_do_not_load_model(
        self,
        load_model,
    ) -> None:
        document = self.make_document(
            "Only result.",
            "document:4",
        )

        reranker = CrossEncoderReranker()

        self.assertEqual(
            reranker.rerank(
                "question",
                [],
            ),
            [],
        )

        self.assertEqual(
            reranker.rerank(
                "question",
                [
                    document,
                ],
            ),
            [
                document,
            ],
        )

        load_model.assert_not_called()

    @patch(
        "rag.reranking._load_cross_encoder"
    )
    def test_model_loading_failure_preserves_rrf_order(
        self,
        load_model,
    ) -> None:
        documents = [
            self.make_document(
                "First.",
                "document:5",
            ),
            self.make_document(
                "Second.",
                "document:6",
            ),
        ]

        load_model.side_effect = RuntimeError(
            "Model download failed."
        )

        reranker = CrossEncoderReranker()

        with self.assertLogs(
            "rag.reranking",
            level="WARNING",
        ):
            results = reranker.rerank(
                "question",
                documents,
            )

        self.assertEqual(
            results,
            documents,
        )

    @patch(
        "rag.reranking._load_cross_encoder"
    )
    def test_prediction_failure_preserves_rrf_order(
        self,
        load_model,
    ) -> None:
        documents = [
            self.make_document(
                "First.",
                "document:7",
            ),
            self.make_document(
                "Second.",
                "document:8",
            ),
        ]

        model = Mock()

        model.predict.side_effect = RuntimeError(
            "Inference failed."
        )

        load_model.return_value = model

        reranker = CrossEncoderReranker()

        with self.assertLogs(
            "rag.reranking",
            level="WARNING",
        ):
            results = reranker.rerank(
                "question",
                documents,
            )

        self.assertEqual(
            results,
            documents,
        )

    @patch(
        "rag.reranking._load_cross_encoder"
    )
    def test_invalid_score_count_preserves_rrf_order(
        self,
        load_model,
    ) -> None:
        documents = [
            self.make_document(
                "First.",
                "document:9",
            ),
            self.make_document(
                "Second.",
                "document:10",
            ),
        ]

        model = Mock()

        model.predict.return_value = [
            0.9,
        ]

        load_model.return_value = model

        reranker = CrossEncoderReranker()

        with self.assertLogs(
            "rag.reranking",
            level="WARNING",
        ):
            results = reranker.rerank(
                "question",
                documents,
            )

        self.assertEqual(
            results,
            documents,
        )

    @patch(
        "rag.reranking._load_cross_encoder"
    )
    def test_equal_scores_preserve_rrf_order(
        self,
        load_model,
    ) -> None:
        documents = [
            self.make_document(
                "First.",
                "document:11",
            ),
            self.make_document(
                "Second.",
                "document:12",
            ),
        ]

        model = Mock()

        model.predict.return_value = [
            0.5,
            0.5,
        ]

        load_model.return_value = model

        reranker = CrossEncoderReranker()

        self.assertEqual(
            reranker.rerank(
                "question",
                documents,
            ),
            documents,
        )

    @patch.dict(
        os.environ,
        {},
        clear=True,
    )
    def test_environment_defaults_enable_multilingual_reranker(
        self,
    ) -> None:
        reranker = CrossEncoderReranker.from_env()

        self.assertIsNotNone(
            reranker
        )

        self.assertEqual(
            reranker.model_name,
            DEFAULT_RERANKER_MODEL,
        )

        self.assertIsNone(
            reranker.device
        )

        self.assertEqual(
            reranker.batch_size,
            16,
        )

    @patch.dict(
        os.environ,
        {
            "RERANKER_ENABLED": "false",
        },
        clear=True,
    )
    def test_reranker_can_be_disabled_from_environment(
        self,
    ) -> None:
        self.assertIsNone(
            CrossEncoderReranker.from_env()
        )

    @patch.dict(
        os.environ,
        {
            "RERANKER_ENABLED": "true",
            "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
            "RERANKER_DEVICE": "cpu",
            "RERANKER_BATCH_SIZE": "4",
        },
        clear=True,
    )
    def test_reranker_reads_custom_environment_settings(
        self,
    ) -> None:
        reranker = CrossEncoderReranker.from_env()

        self.assertIsNotNone(
            reranker
        )

        self.assertEqual(
            reranker.model_name,
            "BAAI/bge-reranker-v2-m3",
        )

        self.assertEqual(
            reranker.device,
            "cpu",
        )

        self.assertEqual(
            reranker.batch_size,
            4,
        )

    @patch.dict(
        os.environ,
        {
            "RERANKER_BATCH_SIZE": "invalid",
        },
        clear=True,
    )
    def test_invalid_environment_batch_size_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            CrossEncoderReranker.from_env()

    def test_invalid_reranker_settings_are_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            CrossEncoderReranker(
                model_name=""
            )

        with self.assertRaises(
            ValueError
        ):
            CrossEncoderReranker(
                batch_size=0
            )