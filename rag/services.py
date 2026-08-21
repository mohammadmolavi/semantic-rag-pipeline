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
    split_document_text,
)

from .llm import OpenRouterClient
from .reranking import CrossEncoderReranker


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


@lru_cache(maxsize=1)
def get_vector_store():
    return build_vector_store(
        get_embedder()
    )

@lru_cache(
    maxsize=1
)
def get_reranker() -> CrossEncoderReranker | None:
    return CrossEncoderReranker.from_env()


def reindex_source(
    source: str,
    text: str,
    previous_chunk_count: int = 0,
) -> int:
    vector_store = get_vector_store()

    if previous_chunk_count:
        delete_source_chunks(
            vector_store,
            source,
            previous_chunk_count,
        )

    if not text.strip():
        return 0

    return index_text(
        vector_store,
        text,
        source,
    )


def delete_source(
    source: str,
    chunk_count: int,
) -> None:
    if chunk_count:
        delete_source_chunks(
            get_vector_store(),
            source,
            chunk_count,
        )


def get_lexical_documents(
    source: str | None = None,
) -> list[Document]:
    from documents.models import (
        Document as StoredDocument,
    )

    stored_documents = (
        StoredDocument.objects
        .filter(
            chunk_count__gt=0,
        )
        .exclude(
            text=""
        )
    )

    if source:
        prefix, separator, document_id = (
            source.partition(":")
        )

        if (
            prefix != "document"
            or not separator
        ):
            return []

        try:
            stored_documents = (
                stored_documents.filter(
                    pk=int(
                        document_id
                    )
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            return []

    lexical_documents: list[Document] = []

    for stored_document in (
        stored_documents.iterator()
    ):
        chunks = split_document_text(
            stored_document.text,
            source=(
                stored_document.vector_source
            ),
        )

        lexical_documents.extend(
            chunks
        )

    return lexical_documents


def ask_question(
    question: str,
    *,
    source: str | None = None,
    top_k: int = 4,
) -> LangChainRagAnswer:
    lexical_documents = (
        get_lexical_documents(
            source
        )
    )

    retriever = build_retriever(
        get_vector_store(),
        source=source,
        top_k=top_k,
        lexical_documents=(
            lexical_documents
        ),
        reranker=get_reranker(),
    )

    pipeline = LangChainRagPipeline(
        retriever,
        OpenRouterClient.from_env(),
    )

    return pipeline.ask(
        question
    )


def serialize_sources(
    documents: list[Document],
) -> list[
    dict[
        str,
        object,
    ]
]:
    return [
        {
            "chunk_index": (
                document.metadata.get(
                    "chunk_index"
                )
            ),
            "source": (
                document.metadata.get(
                    "source"
                )
            ),
            "content": (
                document.page_content
            ),
            "citation": (
                document.metadata.get(
                    "citation"
                )
            ),
        }
        for document in documents
    ]