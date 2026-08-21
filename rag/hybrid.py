"""Hybrid semantic and independent BM25 document retrieval."""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .retrieval import tokenize


class HybridRetriever:
    """Combine vector search with an independent BM25 corpus search."""

    def __init__(
        self,
        vector_store,
        *,
        lexical_documents: list[Document] | None = None,
        source: str | None = None,
        vector_k: int = 12,
        lexical_k: int = 12,
        final_k: int = 4,
    ) -> None:
        self.vector_store = vector_store
        self.lexical_documents = list(lexical_documents or [])
        self.source = source
        self.vector_k = vector_k
        self.lexical_k = lexical_k
        self.final_k = final_k

    def invoke(self, question: str) -> list[Document]:
        vector_documents = self._vector_search(question)

        lexical_documents = self._bm25_search(question)

        ranked_documents = self._merge_scores(
            vector_documents,
            lexical_documents,
        )

        return [
            document
            for document, _ in ranked_documents[: self.final_k]
        ]

    def _vector_search(self, question: str) -> list[Document]:
        search_kwargs = {
            "k": self.vector_k,
        }

        if self.source:
            search_kwargs["filter"] = {
                "source": self.source,
            }

        return self.vector_store.similarity_search(
            question,
            **search_kwargs,
        )

    def _bm25_search(
        self,
        question: str,
    ) -> list[tuple[Document, float]]:
        query_tokens = tokenize(question)

        if not query_tokens:
            return []

        prepared_documents: list[Document] = []

        tokenized_corpus: list[list[str]] = []

        for document in self.lexical_documents:
            document_source = document.metadata.get(
                "source"
            )

            if self.source and document_source != self.source:
                continue

            document_tokens = tokenize(
                document.page_content
            )

            if not document_tokens:
                continue

            prepared_documents.append(
                document
            )

            tokenized_corpus.append(
                document_tokens
            )

        if not prepared_documents:
            return []

        bm25 = BM25Okapi(
            tokenized_corpus
        )

        raw_scores = bm25.get_scores(
            query_tokens
        )

        query_terms = set(
            query_tokens
        )

        matched_documents = [
            (
                document,
                float(score),
            )
            for document, document_tokens, score in zip(
                prepared_documents,
                tokenized_corpus,
                raw_scores,
                strict=True,
            )
            if query_terms.intersection(
                document_tokens
            )
        ]

        matched_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return matched_documents[
            : self.lexical_k
        ]

    def _merge_scores(
        self,
        vector_documents: list[Document],
        lexical_documents: list[
            tuple[Document, float]
        ],
    ) -> list[tuple[Document, float]]:
        scores: dict[
            tuple[object, object],
            float,
        ] = defaultdict(float)

        document_lookup: dict[
            tuple[object, object],
            Document,
        ] = {}

        for rank, document in enumerate(
            vector_documents,
            start=1,
        ):
            key = self._key(
                document
            )

            document_lookup[key] = document

            scores[key] += (
                0.6 / rank
            )

        for rank, (document, _) in enumerate(
            lexical_documents,
            start=1,
        ):
            key = self._key(
                document
            )

            document_lookup.setdefault(
                key,
                document,
            )

            scores[key] += (
                0.4 / rank
            )

        ranked_documents = [
            (
                document_lookup[key],
                score,
            )
            for key, score in scores.items()
        ]

        return sorted(
            ranked_documents,
            key=lambda item: item[1],
            reverse=True,
        )

    @staticmethod
    def _key(
        document: Document,
    ) -> tuple[object, object]:
        return (
            document.metadata.get(
                "source"
            ),
            document.metadata.get(
                "chunk_index"
            ),
        )