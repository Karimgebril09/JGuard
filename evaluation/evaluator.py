from evaluation.refusal.inference.refusal_predictor import RefusalPredictor
from evaluation.harm_detection.llm_judge import SentinelJudge

class Evaluator:
    def __init__(self,type_judge,judge,base_url_judge,api_key_judge):
        self.judge = SentinelJudge(model_type=type_judge,model_name=judge,harm_type="evaluation",\
                                   base_url=base_url_judge, api_key=api_key_judge)
        self.refusal_predictor=RefusalPredictor()

    def evaluate_response(self, llm_response):
        refusal_prediction = self.refusal_predictor.predict(llm_response)
        judge_result = self.judge(llm_response)
        return refusal_prediction or judge_result["output"]
