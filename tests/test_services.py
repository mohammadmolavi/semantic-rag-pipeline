import unittest

from langchain_core.documents import Document

from rag.services import serialize_sources


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