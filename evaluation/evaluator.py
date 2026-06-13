from evaluation.refusal.inference.refusal_predictor import RefusalPredictor
from evaluation.harm_detection.llm_judge import SentinelJudge

class Evaluator:
    def __init__(self,type_judge,judge,base_url_judge,api_key_judge):
        self.judge = SentinelJudge(model_type=type_judge,model_name=judge,harm_type="evaluation",\
                                   base_url=base_url_judge, api_key=api_key_judge)
        self.refusal_predictor=RefusalPredictor()

    def evaluate_response(self, llm_response):
        refusal_prediction = self.refusal_predictor.predict(llm_response)
        # TODO : add fine-tuned harm model 
        fine_tuned_judge_pred = None
        return refusal_prediction or fine_tuned_judge_pred
    
    def evaluate_response_with_feedback(self,user_prompt, llm_response):
        print(f"evaluating response: {llm_response}")
        print(f"user prompt: {user_prompt}")
        refusal_prediction = self.refusal_predictor.predict(llm_response)
        judge_result = self.judge.run(user_prompt, llm_response)
        return refusal_prediction or judge_result["result"], judge_result
