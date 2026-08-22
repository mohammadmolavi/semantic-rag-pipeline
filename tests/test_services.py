import unittest

from unittest.mock import Mock, patch

from langchain_core.documents import Document

from rag.llm import OpenRouterClient
from rag.services import ask_question, serialize_sources


class SourceSerializationTests(
    unittest.TestCase
):
    def test_serialization_preserves_reranked_source_order(
        self,
    ) -> None:
        documents = [
            Document(
                page_content=(
                    "Most relevant chunk."
                ),
                metadata={
                    "source": "document:9",
                    "chunk_index": 4,
                    "citation": "document:9 - chunk 4",
                },
            ),
            Document(
                page_content=(
                    "Less relevant chunk."
                ),
                metadata={
                    "source": "document:1",
                    "chunk_index": 0,
                    "citation": "document:1 - chunk 0",
                },
            ),
        ]

        serialized = serialize_sources(
            documents
        )

        self.assertEqual(
            [
                source["source"]
                for source in serialized
            ],
            [
                "document:9",
                "document:1",
            ],
        )

        self.assertEqual(
            [
                source["chunk_index"]
                for source in serialized
            ],
            [
                4,
                0,
            ],
        )

    def test_serialization_preserves_existing_source_fields(
        self,
    ) -> None:
        document = Document(
            page_content=(
                "Stored document content."
            ),
            metadata={
                "source": "document:3",
                "chunk_index": 2,
                "citation": "document:3 - chunk 2",
            },
        )

        self.assertEqual(
            serialize_sources(
                [
                    document,
                ]
            ),
            [
                {
                    "chunk_index": 2,
                    "source": "document:3",
                    "content": "Stored document content.",
                    "citation": "document:3 - chunk 2",
                },
            ],
        )

    def test_serialization_includes_section_when_available(
        self,
    ) -> None:
        document = Document(
            page_content="Penalty details.",
            metadata={
                "source": "document:4",
                "chunk_index": 1,
                "citation": "document:4 - Contract > Penalties - chunk 1",
                "section_path": "Contract > Penalties",
            },
        )

        serialized = serialize_sources([document])

        self.assertEqual(serialized[0]["section"], "Contract > Penalties")
        self.assertEqual(
            serialized[0]["citation"],
            "document:4 - Contract > Penalties - chunk 1",
        )


class ServiceFailureTests(unittest.TestCase):
    def test_empty_provider_answer_is_rejected(self) -> None:
        client = object.__new__(OpenRouterClient)
        client.model = "fixture-model"
        client._chain = Mock()
        client._chain.invoke.return_value = "  <think>internal</think>  "

        with self.assertRaisesRegex(RuntimeError, "empty answer"):
            client.answer("system", "user")

    def test_unexpected_retrieval_error_is_wrapped_as_runtime_error(self) -> None:
        with (
            patch(
                "rag.services.get_lexical_documents",
                side_effect=OSError("database unavailable"),
            ),
            self.assertLogs("rag.services", level="ERROR"),
            self.assertRaisesRegex(RuntimeError, "temporarily unavailable"),
        ):
            ask_question("test")
