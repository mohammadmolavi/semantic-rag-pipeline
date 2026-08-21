import os

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from .embeddings import SentenceTransformerEmbedder
from .hybrid import HybridRetriever
from .llm import OpenRouterClient


DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://"
    "postgres:postgres@localhost:5432/"
    "roshan_rag"
)

DEFAULT_COLLECTION_NAME = (
    "roshan_rag_chunks"
)


def langchain_database_url(
    database_url: str | None = None,
) -> str:
    url = database_url or os.getenv(
        "DATABASE_URL",
        DEFAULT_DATABASE_URL,
    )

    if url.startswith(
        "postgresql://"
    ):
        return (
            "postgresql+psycopg://"
            + url[
                len("postgresql://"):
            ]
        )

    if url.startswith(
        "postgres://"
    ):
        return (
            "postgresql+psycopg://"
            + url[
                len("postgres://"):
            ]
        )

    return url


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Persian RAG assistant. "
            "Write only the final answer in Persian. "
            "Do not write analysis, thinking, English, "
            "or source JSON. "
            "Use only the provided context. "
            "If the context is not enough, say the "
            "uploaded documents do not contain enough "
            "information. Cite chunk numbers in the "
            "Persian answer, for example (بخش ۰).",
        ),
        (
            "user",
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Write the Persian answer only.",
        ),
    ]
)


@dataclass(frozen=True)
class LangChainRagAnswer:
    answer: str

    sources: list[Document]


def build_vector_store(
    embeddings: SentenceTransformerEmbedder,
    *,
    database_url: str | None = None,
    collection_name: str = (
        DEFAULT_COLLECTION_NAME
    ),
):
    try:
        from langchain_postgres import PGVector

    except ImportError as error:
        raise RuntimeError(
            "Install LangChain PostgreSQL support with: "
            "python -m pip install "
            "'langchain-postgres>=0.0.12,<0.1'"
        ) from error

    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=langchain_database_url(
            database_url
        ),
        use_jsonb=True,
    )


def split_document_text(
    text: str,
    *,
    source: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = splitter.create_documents(
        [text],
        metadatas=[
            {
                "source": source,
                "document_name": Path(
                    source
                ).name,
            }
        ],
    )

    for index, document in enumerate(
        documents
    ):
        document.metadata[
            "chunk_index"
        ] = index

        document_name = document.metadata.get(
            "document_name",
            source,
        )

        document.metadata[
            "citation"
        ] = (
            f"{document_name} - chunk {index}"
        )

    return documents


def chunk_ids_for_source(
    source: str,
    chunk_count: int,
) -> list[str]:
    return [
        f"{source}:{index}"
        for index in range(
            chunk_count
        )
    ]


def add_documents_to_vector_store(
    vector_store,
    documents: list[Document],
) -> None:
    if not documents:
        return

    ids = [
        (
            f"{Path(str(document.metadata.get('source', 'unknown'))).as_posix()}"
            f":{document.metadata['chunk_index']}"
        )
        for document in documents
    ]

    vector_store.add_documents(
        documents,
        ids=ids,
    )


def delete_source_chunks(
    vector_store,
    source: str,
    chunk_count: int,
) -> None:
    ids = chunk_ids_for_source(
        source,
        chunk_count,
    )

    if ids:
        vector_store.delete(
            ids
        )


def index_text(
    vector_store,
    text: str,
    source: str,
) -> int:
    documents = split_document_text(
        text,
        source=source,
    )

    add_documents_to_vector_store(
        vector_store,
        documents,
    )

    return len(
        documents
    )


def dedupe_documents(
    documents: list[Document],
) -> list[Document]:
    unique: list[Document] = []

    seen: set[str] = set()

    for document in documents:
        key = " ".join(
            document.page_content.split()
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            document
        )

    return unique


def build_retriever(
    vector_store,
    *,
    source: str | None = None,
    top_k: int = 4,
    lexical_documents: list[Document] | None = None,
) -> HybridRetriever:
    return HybridRetriever(
        vector_store,
        lexical_documents=lexical_documents,
        source=source,
        vector_k=max(
            top_k * 3,
            12,
        ),
        lexical_k=max(
            top_k * 3,
            12,
        ),
        final_k=top_k,
    )


class LangChainRagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: OpenRouterClient,
    ) -> None:
        self.retriever = retriever

        self.llm = llm

    def ask(
        self,
        question: str,
    ) -> LangChainRagAnswer:
        documents = dedupe_documents(
            self.retriever.invoke(
                question
            )
        )

        prompt_value = PROMPT.invoke(
            {
                "context": (
                    self._format_documents(
                        documents
                    )
                ),
                "question": question,
            }
        )

        messages = prompt_value.to_messages()

        answer = self.llm.answer(
            system_prompt=str(
                messages[0].content
            ),
            user_prompt=str(
                messages[1].content
            ),
        )

        return LangChainRagAnswer(
            answer=answer,
            sources=documents,
        )

    @staticmethod
    def _format_documents(
        documents: list[Document],
    ) -> str:
        if not documents:
            return (
                "No relevant context was found."
            )

        return "\n\n".join(
            (
                "[chunk {chunk_index} "
                "| source: {source}]\n"
                "{content}"
            ).format(
                chunk_index=document.metadata.get(
                    "chunk_index",
                    "unknown",
                ),
                source=document.metadata.get(
                    "source",
                    "unknown",
                ),
                content=document.page_content,
            )
            for document in documents
        )