import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from defenders.tools.rag.src.chunker import Chunk


class Embedder:
    def __init__(self, model_name: str = "tfidf-svd-128", n_components: int = 128):
        self.n_components = n_components
        self.dim = n_components
        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_features=8000
        )
        self._svd = None
        self._fitted = False

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        texts = [c.text for c in chunks]

        if not texts:
            raise ValueError("No chunks provided.")
        print (f"[embedder] Embedding {len(texts)} chunks using TF-IDF and SVD...")
        tfidf = self._vectorizer.fit_transform(texts)   

        n_samples = tfidf.shape[0]
        n_features = tfidf.shape[1]

        max_k = min(n_samples - 1, n_features - 1)

        if max_k < 1:
            vecs = tfidf.toarray().astype(np.float32)
            self.dim = vecs.shape[1]
            self._svd = None
            self._fitted = True
            return normalize(vecs, norm="l2")

        actual_k = min(self.n_components, max_k)

        self._svd = TruncatedSVD(
            n_components=actual_k,
            random_state=42
        )

        vecs = self._svd.fit_transform(tfidf)

        self.dim = vecs.shape[1]
        self._fitted = True

        return normalize(vecs, norm="l2").astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call embed_chunks() first.")

        tfidf = self._vectorizer.transform([query])

        if self._svd is None:
            return normalize(
                tfidf.toarray(),
                norm="l2"
            ).astype(np.float32)

        vecs = self._svd.transform(tfidf)

        return normalize(
            vecs,
            norm="l2"
        ).astype(np.float32)

    def transform_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call embed_chunks() first.")

        texts = [c.text for c in chunks]
        if not texts:
            raise ValueError("No chunks provided.")

        tfidf = self._vectorizer.transform(texts)

        if self._svd is None:
            return normalize(tfidf.toarray(), norm="l2").astype(np.float32)

        vecs = self._svd.transform(tfidf)
        return normalize(vecs, norm="l2").astype(np.float32)