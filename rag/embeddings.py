from dataclasses import dataclass

from langchain_core.embeddings import Embeddings


@dataclass
class SentenceTransformerEmbedder(Embeddings):
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install sentence-transformers to generate retrieval embeddings."
            ) from error

        self._model = SentenceTransformer(self.model_name)
        dimension = self._model.get_sentence_embedding_dimension()
        if not dimension:
            raise RuntimeError(f"Could not detect embedding dimension for {self.model_name}.")
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [embedding.astype(float).tolist() for embedding in embeddings]
