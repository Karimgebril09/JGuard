

import datetime
import json
import os


class MultiAgentEvaluator:
    def __init__(self,evaluator):
        self.evaluator = evaluator

    def evaluate_and_log( self, node_name: str, user_message: str, ai_content: str):
        try:
            result, output = self.evaluator.evaluate_response_with_feedback(user_message, ai_content)
            # print(f"Evaluation result: {result}, output: {output}")
            log_entry = {
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "node": node_name,
                "msg_type": "ai",
                "content": ai_content,
                "judge_run": True,
                "harm_score": int(bool(result)) if isinstance(result, bool) else int(result),
                "reason": output.get("reason"),
            }
        except Exception as e:
            log_entry = {
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "node": node_name,
                "content": ai_content,
                "judge_run": False,
                "harm_score": None,
                "reason": f"Evaluation failed: {e}",
            }

        file_path = "outputs/multi_agent_safety_logs.json"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        logs = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        logs.append(log_entry)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False) 