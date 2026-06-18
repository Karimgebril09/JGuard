import os
import sys

_INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)

import joblib
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from ssm_feature_extractor import StateFeatureExtractor
from tca_feature_extraction import TCAFeatures
from sklearn.preprocessing import PowerTransformer

from transforms import TRANSFORMS
import warnings
warnings.filterwarnings("ignore")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, "..", "models")


class MultiTurnDefender:
    def __init__(self):
        # self._model=joblib.load("./../models/xgb_model_best_result_fixed.pkl")
        self._model=joblib.load(os.path.join(_MODELS_DIR, "xgb_model_best_result_fixed.pkl"))
        self._embedding_model=SentenceTransformer("all-mpnet-base-v2")
        self._ssm_feature_extractor=StateFeatureExtractor(self._embedding_model)
        self._tca_feature_extractor=TCAFeatures(self._embedding_model)

        # feature_info_path="./../config/feature_info.json"
        feature_info_path=os.path.join(_BASE_DIR, "..", "config", "feature_info.json")
        with open(feature_info_path) as f:
            feature_info=json.load(f)
        self.selected_features=feature_info["selected_features"]




    def predict(self, prompt, response) :
        ssm_features, vectors = self._ssm_feature_extractor.extract_features(prompt)
        vectors=pd.DataFrame(vectors, columns=[str(i) for i in range(vectors.shape[1])])
        tca_features = self._tca_feature_extractor.feature_extract(prompt, response)
       
        all_features = pd.concat([tca_features, vectors], axis=1)
        prediction = self._model.predict(all_features)
        return prediction[0]


if __name__ == "__main__":
    defender = MultiTurnDefender()
    prompt = "What is the capital of France?"
    response = "The capital of France is Paris."
    prediction = defender.predict(prompt, response)
    print("Prediction:", prediction)