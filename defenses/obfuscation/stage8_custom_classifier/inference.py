from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import joblib


class TfidfLogRegStage8Classifier:
    def __init__(
        self,
        *,
        vectorizer_path: Path,
        binary_classifier_path: Path,
        binary_encoder_path: Path,
        unsafe_category_classifier_path: Path,
        unsafe_category_encoder_path: Path,
        metadata_path: Path | None = None,
    ) -> None:
        self.vectorizer = joblib.load(vectorizer_path)
        self.binary_classifier = joblib.load(binary_classifier_path)
        self.binary_encoder = joblib.load(binary_encoder_path)
        self.unsafe_category_classifier = joblib.load(unsafe_category_classifier_path)
        self.unsafe_category_encoder = joblib.load(unsafe_category_encoder_path)
        self.metadata: dict[str, Any] = {}
        if metadata_path is not None and metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    def classify(self, text: str) -> dict[str, Any]:
        features = self.vectorizer.transform([text])
        binary_label_id = int(self.binary_classifier.predict(features)[0])
        binary_label = str(self.binary_encoder.inverse_transform([binary_label_id])[0])
        is_safe = binary_label == "safe"

        harmfulness_category = None
        if not is_safe:
            category_label_id = int(self.unsafe_category_classifier.predict(features)[0])
            harmfulness_category = str(
                self.unsafe_category_encoder.inverse_transform([category_label_id])[0]
            )

        binary_confidence = None
        if hasattr(self.binary_classifier, "predict_proba"):
            binary_confidence = float(self.binary_classifier.predict_proba(features).max())

        category_confidence = None
        if not is_safe and hasattr(self.unsafe_category_classifier, "predict_proba"):
            category_confidence = float(
                self.unsafe_category_classifier.predict_proba(features).max()
            )

        mapped_categories = [] if is_safe else [str(harmfulness_category)]

        raw_response = {
            "binary_label": binary_label,
            "is_safe": is_safe,
            "harmfulness_category": harmfulness_category,
            "binary_confidence": binary_confidence,
            "category_confidence": category_confidence,
        }

        return {
            "is_safe": is_safe,
            "raw_categories": [] if is_safe else [str(harmfulness_category)],
            "mapped_categories": mapped_categories,
            "raw_response": json.dumps(raw_response, ensure_ascii=True),
            "confidence": binary_confidence,
            "binary_confidence": binary_confidence,
            "category_confidence": category_confidence,
            "classifier_type": "tfidf_logreg_binary_plus_category",
        }


class DistilBertStage8Classifier:
    def __init__(
        self,
        *,
        tokenizer_dir: Path,
        binary_model_dir: Path,
        binary_encoder_path: Path,
        unsafe_category_model_dir: Path,
        unsafe_category_encoder_path: Path,
        metadata_path: Path | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.binary_encoder = joblib.load(binary_encoder_path)
        self.unsafe_category_encoder = joblib.load(unsafe_category_encoder_path)
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.binary_model = AutoModelForSequenceClassification.from_pretrained(binary_model_dir)
        self.unsafe_category_model = AutoModelForSequenceClassification.from_pretrained(
            unsafe_category_model_dir
        )

        self.device = self._torch.device(
            "cuda" if self._torch.cuda.is_available() else "cpu"
        )
        self.binary_model.to(self.device)
        self.unsafe_category_model.to(self.device)
        self.binary_model.eval()
        self.unsafe_category_model.eval()

        self.metadata: dict[str, Any] = {}
        if metadata_path is not None and metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        configured_max_length = self.metadata.get("max_length")
        self.max_length = int(configured_max_length) if isinstance(configured_max_length, (int, float)) else 128

    def _predict_label_and_confidence(self, model: Any, text: str) -> tuple[int, float]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}

        with self._torch.no_grad():
            logits = model(**encoded).logits
            probs = self._torch.softmax(logits, dim=-1)[0]
            label_index = int(self._torch.argmax(probs).item())
            confidence = float(probs[label_index].item())
        return label_index, confidence

    def classify(self, text: str) -> dict[str, Any]:
        binary_label_id, binary_confidence = self._predict_label_and_confidence(
            self.binary_model, text
        )
        binary_label = str(self.binary_encoder.inverse_transform([binary_label_id])[0])
        is_safe = binary_label == "safe"

        harmfulness_category = None
        category_confidence = None
        if not is_safe:
            category_label_id, category_confidence = self._predict_label_and_confidence(
                self.unsafe_category_model, text
            )
            harmfulness_category = str(
                self.unsafe_category_encoder.inverse_transform([category_label_id])[0]
            )

        raw_response = {
            "binary_label": binary_label,
            "is_safe": is_safe,
            "harmfulness_category": harmfulness_category,
            "binary_confidence": binary_confidence,
            "category_confidence": category_confidence,
        }

        mapped_categories = [] if is_safe else [str(harmfulness_category)]

        return {
            "is_safe": is_safe,
            "raw_categories": [] if is_safe else [str(harmfulness_category)],
            "mapped_categories": mapped_categories,
            "raw_response": json.dumps(raw_response, ensure_ascii=True),
            "confidence": binary_confidence,
            "binary_confidence": binary_confidence,
            "category_confidence": category_confidence,
            "classifier_type": "distilbert_binary_plus_category",
            "model_name": str(self.metadata.get("model_name", "distilbert_finetuned")),
        }


def load_stage8_custom_classifier(
    artifacts_dir: str | Path,
) -> Callable[[str], dict[str, Any]]:
    artifacts = Path(artifacts_dir)
    classifier = TfidfLogRegStage8Classifier(
        vectorizer_path=artifacts / "tfidf_vectorizer.joblib",
        binary_classifier_path=artifacts / "binary_classifier.joblib",
        binary_encoder_path=artifacts / "binary_label_encoder.joblib",
        unsafe_category_classifier_path=artifacts / "unsafe_category_classifier.joblib",
        unsafe_category_encoder_path=artifacts / "unsafe_category_label_encoder.joblib",
        metadata_path=artifacts / "metadata.json",
    )
    return classifier.classify


def load_stage8_distilbert_classifier(
    artifacts_dir: str | Path,
) -> Callable[[str], dict[str, Any]]:
    artifacts = Path(artifacts_dir)
    classifier = DistilBertStage8Classifier(
        tokenizer_dir=artifacts / "tokenizer",
        binary_model_dir=artifacts / "binary_final",
        binary_encoder_path=artifacts / "binary_label_encoder.pkl",
        unsafe_category_model_dir=artifacts / "unsafe_category_final",
        unsafe_category_encoder_path=artifacts / "unsafe_category_label_encoder.pkl",
        metadata_path=artifacts / "metadata.json",
    )
    return classifier.classify
