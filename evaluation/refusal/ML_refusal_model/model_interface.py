from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Dict, cast

import numpy as np
import pandas as pd
import xgboost as xgb
from sentence_transformers import SentenceTransformer

import pipeline


class RefusalClassifierInterface:
    def __init__(
        self,
        model_path: str | Path = "models/xgb_refusal_model.json",
        bundle_path: str | Path = "models/xgb_refusal_bundle.pkl",
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        self.model_path = (base_dir / model_path).resolve()
        self.bundle_path = (base_dir / bundle_path).resolve()

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))

        self.bundle: Dict[str, Any] = {}
        if self.bundle_path.exists():
            with open(self.bundle_path, "rb") as f:
                self.bundle = pickle.load(f)

        self.threshold = float(self.bundle.get("decision_threshold", 0.5))

        self._prepare_preprocessors()

    def _prepare_preprocessors(self) -> None:
        scaler = self.bundle.get("engineered_scaler")
        if scaler is None:
            raise ValueError(
                f"Missing 'engineered_scaler' in bundle at {self.bundle_path}. "
                "Artifacts must include a pre-fitted scaler for inference."
            )
        self.engineered_scaler = scaler

        tfidf_vectorizer = self.bundle.get("tfidf_vectorizer")
        if tfidf_vectorizer is None:
            raise ValueError(
                f"Missing 'tfidf_vectorizer' in bundle at {self.bundle_path}. "
                "Artifacts must include a pre-fitted TF-IDF vectorizer for inference."
            )

        count_vectorizer = self.bundle.get("count_vectorizer")
        if count_vectorizer is None:
            raise ValueError(
                f"Missing 'count_vectorizer' in bundle at {self.bundle_path}. "
                "Artifacts must include a pre-fitted CountVectorizer for inference."
            )

        assert tfidf_vectorizer is not None
        assert count_vectorizer is not None
        self.tfidf_vectorizer = tfidf_vectorizer
        self.count_vectorizer = count_vectorizer

        engineered_dim = int(
            getattr(
                self.engineered_scaler,
                "n_features_in_",
                self._extract_engineered_features(pd.Series([""])).shape[1],
            )
        )
        tfidf_dim = len(tfidf_vectorizer.get_feature_names_out())
        count_dim = len(count_vectorizer.get_feature_names_out())
        self.embedding_dim = int(self.model.n_features_in_) - (engineered_dim + tfidf_dim + count_dim)

        if self.embedding_dim < 0:
            raise ValueError(
                "Computed negative embedding dimension. Check artifact compatibility between "
                "model, scaler, and vectorizers."
            )

        self.embedding_model = None
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            self.embedding_model = None

    @staticmethod
    def _extract_engineered_features(text_series: pd.Series) -> pd.DataFrame:
        return pd.concat(
            [
                text_series.apply(pipeline.extract_length_features).apply(pd.Series),
                text_series.apply(pipeline.detect_refusal_keywords).apply(pd.Series),
                text_series.apply(pipeline.extract_sentiment_features).apply(pd.Series),
                text_series.apply(pipeline.extract_structure_features).apply(pd.Series),
                text_series.apply(pipeline.extract_apologetic_features).apply(pd.Series),
                text_series.apply(pipeline.extract_first_person_features).apply(pd.Series),
                text_series.apply(pipeline.extract_hedging_features).apply(pd.Series),
                text_series.apply(pipeline.extract_opening_pattern_features).apply(pd.Series),
                text_series.apply(pipeline.extract_negation_features).apply(pd.Series),
            ],
            axis=1,
        )

    def _build_features(self, text: str) -> np.ndarray:
        text = str(text)
        processed_text, _ = pipeline.preprocess_text(text)

        single_series = pd.Series([text])
        engineered_features = self._extract_engineered_features(single_series)
        engineered_scaled = self.engineered_scaler.transform(engineered_features).astype(np.float32)

        tfidf_matrix = cast(Any, self.tfidf_vectorizer.transform([processed_text]))
        count_matrix = cast(Any, self.count_vectorizer.transform([processed_text]))
        tfidf_features = tfidf_matrix.toarray().astype(np.float32)
        count_features = count_matrix.toarray().astype(np.float32)
        if self.embedding_model is not None:
            embedding_features = np.asarray(
                self.embedding_model.encode([text], show_progress_bar=False),
                dtype=np.float32,
            )
            if embedding_features.shape[1] != self.embedding_dim:
                adjusted = np.zeros((1, self.embedding_dim), dtype=np.float32)
                width = min(self.embedding_dim, embedding_features.shape[1])
                adjusted[:, :width] = embedding_features[:, :width]
                embedding_features = adjusted
        else:
            embedding_features = np.zeros((1, self.embedding_dim), dtype=np.float32)

        features = np.concatenate(
            [engineered_scaled, tfidf_features, count_features, embedding_features],
            axis=1,
        )

        expected_features = int(self.model.n_features_in_)
        if features.shape[1] != expected_features:
            raise ValueError(
                f"Feature size mismatch: got {features.shape[1]}, expected {expected_features}. "
                "Ensure the same training data and preprocessing settings are used."
            )

        return features

    def predict(self, text: str) -> Dict[str, Any]:
        features = self._build_features(text)
        probability = float(self.model.predict_proba(features)[0, 1])
        label = int(probability >= self.threshold)
        class_name = "refusal" if label == 1 else "not_refusal"

        return {
            "label": label,
            "class_name": class_name,
            "probability": probability,
            "threshold": self.threshold,
        }


_INTERFACE: RefusalClassifierInterface | None = None


def _get_interface() -> RefusalClassifierInterface:
    global _INTERFACE
    if _INTERFACE is None:
        _INTERFACE = RefusalClassifierInterface()
    return _INTERFACE


def classify_text(text: str) -> int:
    """Return binary class: 1 = refusal, 0 = not refusal."""
    return int(_get_interface().predict(text)["label"])


def classify_text_with_details(text: str) -> Dict[str, Any]:
    return _get_interface().predict(text)
