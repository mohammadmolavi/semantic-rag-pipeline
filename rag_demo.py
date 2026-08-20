import argparse

from rag import KeywordRetriever, RagPipeline, split_text
from rag.llm import OpenRouterClient
from rag.loaders import load_docx, load_text_file

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def load_document(path: str) -> str:
    if path.lower().endswith(".docx"):
        return load_docx(path)
    return load_text_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first RAG milestone.")
    parser.add_argument("document", help="Path to a .txt or .docx document")
    parser.add_argument("question", help="Question to ask from the document")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    if load_dotenv:
        load_dotenv()
    text = load_document(args.document)
    chunks = split_text(text)
    retriever = KeywordRetriever.from_texts(chunks, source=args.document)
    pipeline = RagPipeline(retriever, OpenRouterClient.from_env())
    result = pipeline.ask(args.question, top_k=args.top_k)

    print(result.answer)
    print("\nSources:")
    for source in result.sources:
        print(f"- chunk {source.index}: score={source.score:.3f}")


if __name__ == "__main__":
    main()
