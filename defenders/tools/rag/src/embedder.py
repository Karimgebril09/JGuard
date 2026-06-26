import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from defenders.tools.rag.src.chunker import Chunk


class Embedder:
    """Embedder for generating sentence embeddings."""
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self._embedder = SentenceTransformer(model_name)
        self.dim = self._embedder.get_sentence_embedding_dimension()
        

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return normalize(vectors, norm="l2").astype(np.float32)

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        return self._encode([chunk.text for chunk in chunks])

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([query])
