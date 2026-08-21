import argparse

from rag.embeddings import SentenceTransformerEmbedder
from rag.langchain_rag import (
    LangChainRagPipeline,
    add_documents_to_vector_store,
    build_retriever,
    build_vector_store,
    split_document_text,
)
from rag.llm import OpenRouterClient
from rag.loaders import load_document

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangChain RAG demo.")
    parser.add_argument("document", help="Path to a .txt or .docx document")
    parser.add_argument("question", help="Question to ask from the document")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model used for retrieval embeddings.",
    )
    args = parser.parse_args()

    if load_dotenv:
        load_dotenv()
    text = load_document(args.document)
    documents = split_document_text(text, source=args.document)
    embeddings = SentenceTransformerEmbedder(args.model)
    vector_store = build_vector_store(embeddings)
    add_documents_to_vector_store(vector_store, documents)
    retriever = build_retriever(
        vector_store,
        source=args.document,
        top_k=args.top_k,
        lexical_documents=documents,
    )
    pipeline = LangChainRagPipeline(retriever, OpenRouterClient.from_env())
    result = pipeline.ask(args.question)

    print(result.answer)
    print("\nSources:")
    for source in result.sources:
        print(
            "- chunk {chunk_index}: {source}".format(
                chunk_index=source.metadata.get("chunk_index", "unknown"),
                source=source.metadata.get("source", "unknown"),
            )
        )


if __name__ == "__main__":
    main()
