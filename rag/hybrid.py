"""Hybrid retrieval + lightweight reranking utilities."""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document


class HybridRetriever:
    """Combine vector similarity and lexical matching, then rerank.

    The class keeps the same interface used by LangChain's retrievers:
    pipeline code can call invoke(question).
    """

    def __init__(
        self,
        vector_store,
        *,
        source: str | None = None,
        vector_k: int = 12,
        final_k: int = 4,
    ) -> None:
        self.vector_store = vector_store
        self.source = source
        self.vector_k = vector_k
        self.final_k = final_k

    def invoke(self, question: str) -> list[Document]:
        vector_docs = self._vector_search(question)
        lexical_docs = self._keyword_search(question, vector_docs)

        ranked = self._merge_scores(vector_docs, lexical_docs)
        return [doc for doc, _ in ranked[: self.final_k]]

    def _vector_search(self, question: str) -> list[Document]:
        kwargs = {"k": self.vector_k}
        if self.source:
            kwargs["filter"] = {"source": self.source}
        return self.vector_store.similarity_search(question, **kwargs)

    def _keyword_search(self, question: str, docs: list[Document]):
        query_terms = set(self._tokens(question))
        scored = []
        for doc in docs:
            text_terms = set(self._tokens(doc.page_content))
            overlap = len(query_terms & text_terms)
            scored.append((doc, overlap / max(len(query_terms), 1)))
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def _merge_scores(self, vector_docs, lexical_docs):
        scores = defaultdict(float)

        for index, doc in enumerate(vector_docs):
            scores[self._key(doc)] += 1 / (index + 1) * 0.6

        for index, (doc, score) in enumerate(lexical_docs):
            scores[self._key(doc)] += score * 0.4

        lookup = {self._key(doc): doc for doc in vector_docs}
        return sorted(
            [(lookup[key], score) for key, score in scores.items() if key in lookup],
            key=lambda item: item[1],
            reverse=True,
        )

    @staticmethod
    def _tokens(text: str):
        return [x.lower() for x in text.split() if x.strip()]

    @staticmethod
    def _key(doc: Document):
        return (
            doc.metadata.get("source"),
            doc.metadata.get("chunk_index"),
        )
