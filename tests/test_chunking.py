import unittest

from langchain_core.documents import Document

from rag.langchain_rag import (
    LangChainRagPipeline,
    split_document_text,
)


class StructuredChunkingTests(
    unittest.TestCase
):
    def test_heading_metadata_is_attached_to_matching_chunks(
        self,
    ) -> None:
        text = (
            "# شرایط قرارداد\n\n"
            "## پرداخت\n\n"
            "مبلغ قرارداد پنجاه میلیون تومان است.\n\n"
            "## جریمه\n\n"
            "جریمه دیرکرد پنج درصد است."
        )

        documents = split_document_text(
            text,
            source="document:7",
        )

        payment_chunk = next(
            document
            for document in documents
            if "پنجاه میلیون" in (
                document.page_content
            )
        )

        penalty_chunk = next(
            document
            for document in documents
            if "پنج درصد" in (
                document.page_content
            )
        )

        self.assertEqual(
            payment_chunk.metadata[
                "section"
            ],
            "شرایط قرارداد",
        )

        self.assertEqual(
            payment_chunk.metadata[
                "subsection"
            ],
            "پرداخت",
        )

        self.assertEqual(
            payment_chunk.metadata[
                "section_path"
            ],
            "شرایط قرارداد > پرداخت",
        )

        self.assertEqual(
            penalty_chunk.metadata[
                "subsection"
            ],
            "جریمه",
        )

        self.assertIn(
            "شرایط قرارداد > جریمه",
            penalty_chunk.metadata[
                "citation"
            ],
        )

    def test_unstructured_text_keeps_previous_metadata_contract(
        self,
    ) -> None:
        documents = split_document_text(
            "Simple document text without headings.",
            source="document:8",
        )

        self.assertEqual(
            len(
                documents
            ),
            1,
        )

        self.assertEqual(
            documents[0].metadata[
                "source"
            ],
            "document:8",
        )

        self.assertEqual(
            documents[0].metadata[
                "chunk_index"
            ],
            0,
        )

        self.assertEqual(
            documents[0].metadata[
                "citation"
            ],
            "document:8 - chunk 0",
        )

        self.assertNotIn(
            "section_path",
            documents[0].metadata,
        )

    def test_empty_document_produces_no_chunks(
        self,
    ) -> None:
        self.assertEqual(
            split_document_text(
                "   ",
                source="document:9",
            ),
            [],
        )

    def test_chunk_indexes_remain_unique_across_sections(
        self,
    ) -> None:
        text = (
            "# First section\n\n"
            "First content.\n\n"
            "# Second section\n\n"
            "Second content."
        )

        documents = split_document_text(
            text,
            source="document:10",
        )

        self.assertEqual(
            [
                document.metadata[
                    "chunk_index"
                ]
                for document in documents
            ],
            list(
                range(
                    len(
                        documents
                    )
                )
            ),
        )

    def test_formatted_context_contains_section_path(
        self,
    ) -> None:
        document = Document(
            page_content=(
                "Penalty is five percent."
            ),
            metadata={
                "source": "document:11",
                "chunk_index": 2,
                "section_path": "Contract > Penalties",
            },
        )

        context = LangChainRagPipeline._format_documents(
            [
                document,
            ]
        )

        self.assertIn(
            "section: Contract > Penalties",
            context,
        )

        self.assertIn(
            "Penalty is five percent.",
            context,
        )

    def test_formatted_context_without_section_keeps_previous_shape(
        self,
    ) -> None:
        document = Document(
            page_content=(
                "Simple content."
            ),
            metadata={
                "source": "document:12",
                "chunk_index": 0,
            },
        )

        context = LangChainRagPipeline._format_documents(
            [
                document,
            ]
        )

        self.assertIn(
            "[chunk 0 | source: document:12]",
            context,
        )

        self.assertNotIn(
            "section:",
            context,
        )