import joblib
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from ssm_feature_extractor import StateFeatureExtractor
from tca_feature_extraction import TCAFeatures
from sklearn.preprocessing import PowerTransformer

from transforms import TRANSFORMS


class MultiTurnDefender:
    def __init__(self):
        self._model=joblib.load("./../models/xgb_model_best_result_fixed.pkl")
        self._embedding_model=SentenceTransformer("all-mpnet-base-v2")
        self._ssm_feature_extractor=StateFeatureExtractor(self._embedding_model)
        self._tca_feature_extractor=TCAFeatures(self._embedding_model)

        feature_info_path="./../config/feature_info.json"
        with open(feature_info_path) as f:
            feature_info=json.load(f)
        self.selected_features=feature_info["selected_features"]

    def _apply_transform(self, series, transform):
        if transform == "log1p":
            return np.log1p(np.maximum(series, 0))

        if transform == "square":
            return np.square(series)

        if transform == "binarize":
            return (series > 0).astype(float)

        if transform == "yeo-johnson":
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            return pt.fit_transform(
                np.asarray(series).reshape(-1, 1)
            ).flatten()

        return series

    def apply_transforms(self, row) :
        for feature, transform in TRANSFORMS.items():
            if feature not in row:
                continue
            s=pd.Series([row[feature]])
            row[f"{feature}"]=float(self._apply_transform(s, transform).iloc[0])
        return row


    def predict(self, prompt, response) :
        ssm_features, vectors = self._ssm_feature_extractor.extract_features(prompt)
        vectors=pd.DataFrame(vectors, columns=[str(i) for i in range(vectors.shape[1])])
        tca_features = pd.DataFrame([self._tca_feature_extractor.feature_extract(prompt, response)])
        features = pd.concat([ssm_features, tca_features], axis=1)
        features = features[self.selected_features]
        # transformed_features = self.apply_transforms(features)
        all_features = pd.concat([features, vectors], axis=1)
        prediction = self._model.predict(all_features)
        return prediction[0]


if __name__ == "__main__":
    defender = MultiTurnDefender()
    prompt = "What is the capital of France?"
    response = "The capital of France is Paris."
    prediction = defender.predict(prompt, response)
    print("Prediction:", prediction)