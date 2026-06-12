from __future__ import annotations

import os
from typing import Any

import joblib
from sentence_transformers import SentenceTransformer

# Default model path using absolute path resolution
_DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "models", "svm_defender_model.joblib")
)


class RolePlayingDefender:
    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        """
        Initialize the Defender Pipeline by loading the saved components and the embedding model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Please ensure the model is trained and saved."
            )

        self.components = joblib.load(model_path)
        self.classifier = self.components["classifier"]
        self.pca = self.components["pca"]
        self.scaler = self.components["scaler"]
        self.label_encoder = self.components["label_encoder"]

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def predict(self, prompt: str) -> dict[str, Any]:
        """
        Perform all preprocessing steps and return the model's prediction.
        """
        # 1. Generate Embedding
        embedding = self.embedding_model.encode([str(prompt)])

        # 2. Scale
        scaled_embedding = self.scaler.transform(embedding)

        # 3. PCA
        pca_embedding = self.pca.transform(scaled_embedding)

        # 4. Predict class
        prediction = self.classifier.predict(pca_embedding)[0]

        # 5. Get probability if available
        probability = None
        if hasattr(self.classifier, "predict_proba"):
            probability = self.classifier.predict_proba(pca_embedding)[0].tolist()

        # 6. Decode label
        label = prediction
        if self.label_encoder and hasattr(self.label_encoder, "classes_"):
            label = self.label_encoder.inverse_transform([prediction])[0]
        else:
            # Fallback if label_encoder is not fitted or missing
            # In many binary classifiers, 0 is benign and 1 is attack
            if prediction == 0:
                label = "benign"
            elif prediction == 1:
                label = "attack"

        # 7. Determine safety using the raw prediction
        is_safe = bool(prediction == 0)

        return {
            "label": label,
            "prediction_raw": int(prediction) if hasattr(prediction, "__int__") else prediction,
            "probability": probability,
            "is_safe": is_safe,
        }


_cached_defender: RolePlayingDefender | None = None


def get_role_playing_defender(model_path: str | None = None) -> RolePlayingDefender:
    global _cached_defender
    if _cached_defender is None:
        _cached_defender = RolePlayingDefender(model_path or _DEFAULT_MODEL_PATH)
    return _cached_defender


def run_role_playing_pipeline(
    raw_input: str,
    *,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Runs the role-playing defense pipeline."""
    defender = get_role_playing_defender(model_path)
    result = defender.predict(raw_input)

    is_safe = result["is_safe"]

    return {
        "is_safe": is_safe,
        "action": "block" if not is_safe else "allow",
    }


def run_role_playing_guard(
    raw_input: str,
    *,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Run the pipeline and return a minimal integration-friendly result."""
    return run_role_playing_pipeline(
        raw_input,
        model_path=model_path,
    )


# if __name__ == "__main__":
#     import sys

#     # Default test prompt
#     test_prompt = "I want to know how to hack a website"
#     if len(sys.argv) > 1:
#         test_prompt = " ".join(sys.argv[1:])

#     print(f"Running role-playing guard for: '{test_prompt}'")
#     try:
#         res = run_role_playing_guard(test_prompt)
#         print(res)
#     except Exception as e:
#         print(f"Error: {e}")
