from pathlib import Path

from defenders.obfuscation.stage8_custom_classifier import load_stage8_distilbert_classifier
from evaluation.refusal.inference.refusal_predictor import RefusalPredictor
from evaluation.harm_detection.llm_judge import SentinelJudge

class Evaluator:
    def __init__(self,type_judge,judge,base_url_judge,api_key_judge):
        self.judge = SentinelJudge(model_type=type_judge,model_name=judge,harm_type="evaluation",\
                                   base_url=base_url_judge, api_key=api_key_judge)
        self.refusal_predictor=RefusalPredictor()
        artifacts_dir = (
            Path(__file__).resolve().parents[1]
            / "defenders"
            / "obfuscation"
            / "stage8_custom_classifier"
            / "distilbert_artifacts_e4"
        )
        self.fine_tuned_harm_classifier = load_stage8_distilbert_classifier(artifacts_dir)

    def evaluate_response(self, llm_response):
        refusal_prediction = self.refusal_predictor.predict(llm_response)
        fine_tuned_judge_result = self.fine_tuned_harm_classifier(llm_response)
        fine_tuned_judge_pred = not bool(fine_tuned_judge_result.get("is_safe", True))
        return refusal_prediction or fine_tuned_judge_pred
    
    def evaluate_response_with_feedback(self,user_prompt, llm_response):
        print(f"evaluating response: {llm_response}")
        print(f"user prompt: {user_prompt}")
        refusal_prediction = self.refusal_predictor.predict(llm_response)
        judge_result = self.judge.run(user_prompt, llm_response)
        return refusal_prediction or judge_result["result"], judge_result
