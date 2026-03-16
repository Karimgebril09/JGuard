import json
import datetime
from langgraph.checkpoint.memory import MemorySaver
import os
import hashlib


class SafetyCheckpointer(MemorySaver):
    def __init__(self, evaluator):
        super().__init__()
        self.evaluator = evaluator
        self.seen_ai_messages = set()   # prevent duplicate evaluation

    def put(self, config, checkpoint, metadata, new_versions):

        #---------------------------------------------
        writes = metadata.get("writes")
        node_name = list(writes.keys())[0] if isinstance(writes, dict) and writes else metadata.get("source", "system")

        state_values = checkpoint.get("channel_values", {})
        messages = state_values.get("messages", [])

        if not messages:
            return super().put(config, checkpoint, metadata, new_versions)

        last_msg = messages[-1]

        msg_text = getattr(last_msg, "content", "")

        log_entry = {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "node": node_name,
            "msg_type": type(last_msg).__name__,
            "content": msg_text,
            "judge_run": False,
            "harm_score": None,
            "reason": None
        }

        if getattr(last_msg, "type", None) == "ai":

            if getattr(last_msg, "tool_calls", None):
                log_entry["msg_type"] = "ai_tool_call"
                log_entry["content"] = [tc["name"] for tc in last_msg.tool_calls]
                self._append_log(log_entry)
                return super().put(config, checkpoint, metadata, new_versions)
            
            msg_id = last_msg.id 

            if msg_id in self.seen_ai_messages:
                return super().put(config, checkpoint, metadata, new_versions)

            self.seen_ai_messages.add(msg_id)

            user_prompt = next(
                (m.content for m in messages if getattr(m, "type", None) == "human"), ""
            )

            score = self.evaluator.check_message(last_msg, user_prompt)
            log_entry["judge_run"] = True
            log_entry["harm_score"] = score.get("harm")
            log_entry["reason"] = score.get("reason")

        self._append_log(log_entry)
        return super().put(config, checkpoint, metadata, new_versions)

    def _append_log(self, data):
        file = "safety_logs.json"
        logs = []

        if os.path.exists(file):
            with open(file, "r") as f:
                logs = json.load(f)

        logs.append(data)

        with open(file, "w") as f:
            json.dump(logs, f, indent=2)