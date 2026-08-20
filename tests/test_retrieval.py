from rag.retrieval import KeywordRetriever, split_text


def test_split_text_uses_overlap() -> None:
    chunks = split_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=3)

    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]


def test_keyword_retriever_returns_relevant_chunk() -> None:
    retriever = KeywordRetriever.from_texts(
        [
            "Django admin is used for document management.",
            "PostgreSQL stores vectors with pgvector.",
            "OpenRouter connects the app to language models.",
        ],
        source="sample",
    )

    results = retriever.search("How do we store vectors?", top_k=1)

    assert len(results) == 1
    assert results[0].index == 1
