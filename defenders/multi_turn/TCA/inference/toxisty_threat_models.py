import pandas as pd
import joblib
import numpy as np


class BaseClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def _prepare(self, emb):
        return emb.reshape(1, -1) # as it expect a 2D array 

    def score(self, emb):
        x = self._prepare(emb)
        return float(self.model.predict_proba(x)[0, 1])

    def predict(self, emb, threshold=0.5):
        return int(self.score(emb) >= threshold)


class ThreatModel(BaseClassifier):  #logistic needs plain numpy
    def __init__(self):
        super().__init__("defenders/multi_turn/integrated/models/threat_lr.joblib")


class ToxicityModel(BaseClassifier):  #lightGBM named DataFrame
    def __init__(self):
        super().__init__("defenders/multi_turn/integrated/models/toxicity_lgb.joblib")

    def _prepare(self, emb):
        arr = emb.reshape(1, -1)
        return pd.DataFrame(arr, columns=[f"Column_{i}" for i in range(arr.shape[1])])