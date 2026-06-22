import joblib
import numpy as np


class BaseClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def _prepare(self, emb):
        #sklearn models expect 2D arr so reshape it and also float faster
        emb = np.asarray(emb, dtype=np.float32)
        return emb.reshape(1, -1)

    def score(self, emb):
        x = self._prepare(emb)
        return float(self.model.predict_proba(x)[0, 1])

    def predict(self, emb, threshold=0.5):
        return int(self.score(emb) >= threshold)


class ThreatModel(BaseClassifier):
    #easy to call and get score for feature extraction
    def __init__(self):
        super().__init__("defenders/multi_turn/integrated/models/threat_lr.joblib")


class ToxicityModel(BaseClassifier):
    def __init__(self):
        super().__init__("defenders/multi_turn/integrated/models/toxicity_lgb.joblib")