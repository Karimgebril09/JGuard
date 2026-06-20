from evaluation.harm_detection.llm_judge import SentinelJudge


class JailbreakEvaluator:
    def __init__(self):
        self._judge = SentinelJudge(
            model_type="ollama",
            model_name="qwen2.5:3b-instruct",
            harm_type="evaluation",
        )

    def check_message(self, message, prompt):
        ai_content = message.content if hasattr(message, "content") else str(message)
        user_prompt = prompt.content if hasattr(prompt, "content") else str(prompt)

        output = self._judge.run(user_prompt=user_prompt, llm_response=ai_content)

        print(f"Evaluating: '{ai_content[:50]}...' -> Harm: {output['result']}, Reason: {output['reason']}")

        return {"harm": output["result"], "reason": output["reason"]}
