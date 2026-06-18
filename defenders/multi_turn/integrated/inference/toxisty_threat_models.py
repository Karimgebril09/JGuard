
import os
import sys

import joblib
import numpy as np

class ThreatModel:
    def __init__(self):
        self.model = joblib.load( "defenders/multi_turn/integrated/models/threat_lr.joblib")
    def score(self, emb):
        emb = np.asarray(emb, dtype=np.float32)
        input = emb.reshape(1, -1)
        return float(self.model.predict_proba(input)[0, 1])
    def predict(self, emb, threshold=0.5):
        return int(self.score(emb) >= threshold)
    
class ToxicityModel:
    def __init__(self):
        self.model = joblib.load( "defenders/multi_turn/integrated/models/toxicity_lgb.joblib")
    def score(self, emb):
        emb = np.asarray(emb, dtype=np.float32)
        input = emb.reshape(1, -1)
        return float(self.model.predict_proba(input)[0, 1])

    def predict(self, emb, threshold = 0.5) :
        return int(self.score(emb) >= threshold)
