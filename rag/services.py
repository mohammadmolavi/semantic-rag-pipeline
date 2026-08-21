from functools import lru_cache

from langchain_core.documents import Document

from .embeddings import SentenceTransformerEmbedder
from .langchain_rag import (
    LangChainRagAnswer,
    LangChainRagPipeline,
    build_retriever,
    build_vector_store,
    delete_source_chunks,
    index_text,
)
from .llm import OpenRouterClient


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


@lru_cache(maxsize=1)
def get_vector_store():
    return build_vector_store(get_embedder())


def reindex_source(source: str, text: str, previous_chunk_count: int = 0) -> int:
    vector_store = get_vector_store()
    if previous_chunk_count:
        delete_source_chunks(vector_store, source, previous_chunk_count)
    if not text.strip():
        return 0
    return index_text(vector_store, text, source)


def delete_source(source: str, chunk_count: int) -> None:
    if chunk_count:
        delete_source_chunks(get_vector_store(), source, chunk_count)


def ask_question(
    question: str,
    *,
    source: str | None = None,
    top_k: int = 4,
) -> LangChainRagAnswer:
    fetch_k = top_k if source else max(top_k * 2, 8)
    retriever = build_retriever(get_vector_store(), source=source, top_k=fetch_k)
    pipeline = LangChainRagPipeline(retriever, OpenRouterClient.from_env())
    return pipeline.ask(question)


def serialize_sources(documents: list[Document]) -> list[dict[str, object]]:
    ordered_documents = sorted(
        documents,
        key=lambda document: (
            str(document.metadata.get("source", "")),
            int(document.metadata.get("chunk_index", 0)),
        ),
    )

    return [
        {
            "chunk_index": document.metadata.get("chunk_index"),
            "source": document.metadata.get("source"),
            "content": document.page_content,
            "citation": document.metadata.get("citation"),
        }
        for document in ordered_documents
    ]
