import unittest

from rag.retrieval import (
    KeywordRetriever,
    normalize_text,
    split_text,
    tokenize,
)


class RetrievalTests(
    unittest.TestCase
):
    def test_split_text_uses_overlap(
        self,
    ) -> None:
        chunks = split_text(
            "abcdefghijklmnopqrstuvwxyz",
            chunk_size=10,
            overlap=3,
        )

        self.assertEqual(
            chunks,
            [
                "abcdefghij",
                "hijklmnopq",
                "opqrstuvwx",
                "vwxyz",
            ],
        )

    def test_keyword_retriever_returns_relevant_chunk(
        self,
    ) -> None:
        retriever = KeywordRetriever.from_texts(
            [
                "Django admin is used for document management.",
                "PostgreSQL stores vectors with pgvector.",
                "OpenRouter connects the app to language models.",
            ],
            source="sample",
        )

        results = retriever.search(
            "How do we store vectors?",
            top_k=1,
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].index,
            1,
        )

    def test_normalize_text_unifies_arabic_and_persian_characters(
        self,
    ) -> None:
        self.assertEqual(
            normalize_text(
                "كتاب عربي"
            ),
            "کتاب عربی",
        )

    def test_normalize_text_unifies_persian_and_arabic_digits(
        self,
    ) -> None:
        self.assertEqual(
            normalize_text(
                "فاکتور ۱۲۳ و ٤٥٦"
            ),
            "فاکتور 123 و 456",
        )

    def test_normalize_text_removes_diacritics_and_tatweel(
        self,
    ) -> None:
        self.assertEqual(
            normalize_text(
                "سَــــلامْ"
            ),
            "سلام",
        )

    def test_normalize_text_expands_unicode_presentation_forms(
        self,
    ) -> None:
        self.assertEqual(
            normalize_text(
                "ﻻ"
            ),
            "لا",
        )

    def test_normalize_text_removes_directional_controls(
        self,
    ) -> None:
        self.assertEqual(
            normalize_text(
                "س\u200fلام"
            ),
            "سلام",
        )

    def test_tokenize_treats_half_space_like_regular_space(
        self,
    ) -> None:
        self.assertEqual(
            tokenize(
                "می\u200cرود"
            ),
            tokenize(
                "می رود"
            ),
        )

        self.assertEqual(
            tokenize(
                "می\u200cرود"
            ),
            [
                "می",
                "رود",
            ],
        )

    def test_tokenize_normalizes_english_case_and_persian_digits(
        self,
    ) -> None:
        self.assertEqual(
            tokenize(
                "NeRF Version-۲"
            ),
            [
                "nerf",
                "version",
                "2",
            ],
        )

    def test_keyword_retriever_matches_arabic_persian_variants(
        self,
    ) -> None:
        retriever = KeywordRetriever.from_texts(
            [
                "راهنمای كتابخانه عمومی",
                "تنظیمات شبکه",
            ]
        )

        results = retriever.search(
            "کتابخانه",
            top_k=1,
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].index,
            0,
        )

    def test_keyword_retriever_matches_normalized_digits(
        self,
    ) -> None:
        retriever = KeywordRetriever.from_texts(
            [
                "شماره قرارداد ۴۵۶",
                "شماره قرارداد ۹۸۷",
            ]
        )

        results = retriever.search(
            "456",
            top_k=1,
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].index,
            0,
        )

    def test_split_text_preserves_original_document_content(
        self,
    ) -> None:
        original_text = "كتاب شماره ۱۲۳"

        self.assertEqual(
            split_text(
                original_text
            ),
            [
                original_text,
            ],
        )