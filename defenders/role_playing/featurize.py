from __future__ import annotations
from typing import Iterable
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
MAX_FEATURES = 20_000


def build_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=5,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        lowercase=True,
    )
    return word, char


class TfidfFeaturizer:
    def __init__(self) -> None:
        self.word, self.char = build_vectorizers()

    def fit_transform(self, texts: Iterable[str]) -> csr_matrix:
        word_features = self.word.fit_transform(texts)
        char_features = self.char.fit_transform(texts)
        return hstack([word_features, char_features]).tocsr()

    def transform(self, texts: Iterable[str]) -> csr_matrix:
        word_features = self.word.transform(texts)
        char_features = self.char.transform(texts)
        return hstack([word_features, char_features]).tocsr()

    @property
    def n_features(self) -> int:
        return len(self.word.vocabulary_) + len(self.char.vocabulary_)
