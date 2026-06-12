from evaluation.refusal.inference.fasttext_refusal_model import FastTextRefusalDetector
from evaluation.refusal.inference.predict_dl_bert import RefusalInference
import numpy as np

class RefusalPredictor:
    def __init__(self):
        self.model_dl_static=FastTextRefusalDetector()
        self.model_dl_dynamic=RefusalInference()
        # TODO: initialize ML MODELS HERE
        # TODO: ADD JUDGE

    def predict(self,llm_response):
        pred1=self.model_dl_static.predict(llm_response)
        pred2=self.model_dl_dynamic.predict(llm_response)
        # TODO: Implement judge logic to combine predictions

        return np.mean([pred1, pred2]) > 0.5 

