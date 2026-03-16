import os
from pathlib import Path

from evaluation.harm_detection.llm_judge import SentinelJudge

class JailbreakEvaluator:
    def check_message(self, message, prompt):
       
        ai_content = message.content if hasattr(message, 'content') else str(message)
        
        user_prompt = prompt.content if hasattr(prompt, 'content') else str(prompt)

        
        response = SentinelJudge.invoke({
            "user_prompt": user_prompt,
            "llm_response": ai_content,
            "max_trials": 2,
            "default_val": 0,
            "result": -1,
            "reason": ""
        })

      
        print(f"Evaluating: '{ai_content[:50]}...' -> Harm: {response['result']}, Reason: {response['reason']}")

        return {
            "harm": response["result"],
            "reason": response["reason"]
        }

    # def log_results(self, scores):
    #     print("Safety evaluation scores:", scores)