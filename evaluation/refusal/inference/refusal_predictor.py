from evaluation.refusal.inference.fasttext_refusal_model import FastTextRefusalDetector
from evaluation.refusal.inference.predict_dl_bert import RefusalInference
from evaluation.refusal.ML_refusal_model.model_interface import MLRefusalClassifierInterface
import numpy as np

class RefusalPredictor:
    def __init__(self):
        self.model_dl_static=FastTextRefusalDetector()
        self.model_dl_dynamic=RefusalInference()
        # self.model_ml=MLRefusalClassifierInterface()

    def predict(self,llm_response):
        pred1=self.model_dl_static.predict(llm_response)
        pred2=self.model_dl_dynamic.predict(llm_response)
        # pred3=self.model_ml.predict(llm_response)

        return np.sum([pred1, pred2['label']]) >=1


if __name__ == "__main__":
    predictor = RefusalPredictor()
    response = "I cannot provide that information."
    is_refusal = predictor.predict(response)
    print(f"Is the response a refusal? {is_refusal}")

