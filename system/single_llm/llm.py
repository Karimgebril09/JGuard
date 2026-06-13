from collections import deque
from threading import Lock
from typing import Any, Callable, cast

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from defenders.multi_turn.integrated.inference.multi_turn_defender import MultiTurnDefender
from defenders.obfuscation.pipeline import run_obfuscation
from defenders.pii_detection.src.pii_engine import PIIEngine
from defenders.pii_detection.src.strategies import BlockStrategy, EncryptStrategy, MaskStrategy, PIIStrategy
from defenders.role_playing.pipeline import run_role_playing_guard

load_dotenv("./.env")


def _resolve_pii_strategy(strategy_name: str) -> PIIStrategy:
    normalized = strategy_name.strip().lower()
    if normalized == "encrypt":
        return EncryptStrategy()
    if normalized == "block":
        return BlockStrategy()
    return MaskStrategy()

class LLM:
    def __init__(
        self,
        model_name: str,
        model_type: str,
        base_url: str | None = None,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        api_key: str | None = None,
        history_length: int = 5,
        obfuscation_protection: bool = False,
        roleplay_protection: bool = False,
        multi_turn_protection: bool = False,
        pii_protection: bool = False,
        pii_strategy: str = "mask",
        use_history: bool = True,
    ):
        self.model_name = model_name
        self.model: Any = None
        self.base_url = base_url
        self.temperature = temperature
        self.system_prompt = system_prompt or ""
        self.model_type = model_type
        self.api_key = api_key
        self.use_history = use_history
        self.buffer = deque(maxlen=history_length)

        self.obfuscation_protection = obfuscation_protection
        self.roleplay_protection = roleplay_protection
        self.multi_turn_protection = multi_turn_protection
        self.pii_protection = pii_protection
        self.pii_strategy = pii_strategy

        self.pii_engine = PIIEngine(strategy=_resolve_pii_strategy(self.pii_strategy))
        self.pii_lock = Lock()

        self.multi_turn_defender = MultiTurnDefender() if self.multi_turn_protection else None
        self.multi_turn_lock = Lock()
        self.multi_turn_state: dict[str, Any] = {}
        self.last_response = ""


        if self.model_type == "ollama":
            self.model = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                base_url=self.base_url,
            )

        elif self.model_type == "gemini":
            self.model= ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,   
                google_api_key=self.api_key,
            )

        elif self.model_type == "openai":
            self.model= ChatOpenAI(
                model=self.model_name,
                api_key=cast(Any, self.api_key),
                base_url=self.base_url,
                temperature=self.temperature,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def get_model(self):
        return self.model

    def _prepend_system_prompt(self, messages: Any) -> Any:
        system_prompt = self.system_prompt.strip()
        if not system_prompt:
            return messages

        system_message = {"role": "system", "content": system_prompt}
        if isinstance(messages, list):
            return [system_message, *messages]
        return [system_message, {"role": "user", "content": str(messages)}]
    
    def generate_response(self, prompt):
        messages = self._prepend_system_prompt([{"role": "user", "content": str(prompt)}])
        response = self.model.invoke(messages)
        return response
    
    def generate_response_buffered(self, prompt):
        if not self.use_history:
            return self.generate_response(prompt)

        self.buffer.append(prompt)
        buffered_messages = [{"role": "user", "content": str(item)} for item in self.buffer]
        messages = self._prepend_system_prompt(buffered_messages)
        response = self.model.invoke(messages)
        return response

    def _apply_obfuscation(self, prompt: str) -> tuple[str, str | None, str | None, bool]:
        if not self.obfuscation_protection:
            return prompt, None, None, False

        result = run_obfuscation(prompt)
        clean_prompt = str(result.get("clean_text", prompt))
        decision = str(result.get("decision")) if result.get("decision") is not None else None
        harm_label = str(result.get("harm_label")) if result.get("harm_label") is not None else None
        blocked = not bool(result.get("is_safe", True))
        return clean_prompt, decision, harm_label, blocked

    def _apply_role_playing(self, prompt_text: str) -> bool:
        if not self.roleplay_protection:
            return False

        result = run_role_playing_guard(prompt_text)
        action = str(result.get("action", "")).strip().lower()
        if action == "block":
            return True

        is_safe = bool(result.get("is_safe", True))
        return not is_safe

    def _apply_multi_turn(self, prompt_text: str) -> bool:
        if not self.multi_turn_protection:
            return False

        if self.multi_turn_defender is None:
            raise RuntimeError("Multi-turn defender is not initialized.")

        with self.multi_turn_lock:
            prediction = self.multi_turn_defender.predict(prompt_text, self.last_response)

        prediction_value = int(prediction)
        self.multi_turn_state["last_prediction"] = prediction_value
        return prediction_value == 1

    def _apply_pii(self, prompt_text: str) -> tuple[str, bool]:
        if not self.pii_protection:
            return prompt_text, False

        with self.pii_lock:
            self.pii_engine.set_strategy(_resolve_pii_strategy(self.pii_strategy))
            pii_result = str(self.pii_engine.process(prompt_text))

        if pii_result == "[BLOCKED: PII DETECTED]":
            return pii_result, True
        return pii_result, False

    def _messages_with_history(self, history: list[dict[str, str]], prompt_text: str):
        if not self.use_history:
            return self._prepend_system_prompt(prompt_text)

        prior_messages = history[:-1]
        messages = [
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
            for message in prior_messages
        ]
        messages.append({"role": "user", "content": prompt_text})
        return self._prepend_system_prompt(messages)

    def _extract_text(self, response: Any) -> str:
        if response is None:
            return ""
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                    continue
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            return "".join(chunks)
        return str(response)

    def _call_foundational_model(self, history: list[dict[str, str]], prompt_text: str) -> str:
        messages = self._messages_with_history(history, prompt_text)
        response = self.model.invoke(messages)
        text = self._extract_text(response).strip()
        if text:
            return text
        return "Model returned an empty response."

    def chat_secure(
        self,
        prompt: str,
        history: list[dict[str, str]],
        reply_fn: Callable[[str], str] | None = None,
    ) -> dict[str, str | bool | None]:
        clean_prompt, decision, harm_label, blocked = self._apply_obfuscation(prompt)
        if blocked:
            blocked_reply = "Request blocked by harm detector."
            self.last_response = blocked_reply
            self.multi_turn_state["last_response"] = blocked_reply
            return {
                "reply": blocked_reply,
                "blocked": True,
                "triggered_defense": "obfuscation and preprocessing",
                "decision": decision,
                "harm_label": harm_label,
            }

        roleplay_blocked = self._apply_role_playing(clean_prompt)
        if roleplay_blocked:
            blocked_reply = "Request blocked by role-playing defender."
            self.last_response = blocked_reply
            self.multi_turn_state["last_response"] = blocked_reply
            return {
                "reply": blocked_reply,
                "blocked": True,
                "triggered_defense": "roleplay",
                "decision": "unsafe",
                "harm_label": None,
            }

        multi_turn_blocked = self._apply_multi_turn(clean_prompt)
        if multi_turn_blocked:
            blocked_reply = "Request blocked by multi-turn defender."
            self.last_response = blocked_reply
            self.multi_turn_state["last_response"] = blocked_reply
            return {
                "reply": blocked_reply,
                "blocked": True,
                "triggered_defense": "multi_turn",
                "decision": "unsafe",
                "harm_label": None,
            }

        pii_prompt, pii_blocked = self._apply_pii(clean_prompt)
        if pii_blocked:
            blocked_reply = "Request blocked by pii model."
            self.last_response = blocked_reply
            self.multi_turn_state["last_response"] = blocked_reply
            return {
                "reply": blocked_reply,
                "blocked": True,
                "triggered_defense": "pii",
                "decision": "unsafe",
                "harm_label": None,
            }

        if reply_fn is not None:
            reply = reply_fn(pii_prompt)
        else:
            reply = self._call_foundational_model(history=history, prompt_text=pii_prompt)

        self.last_response = reply
        self.multi_turn_state["last_response"] = reply
        return {
            "reply": reply,
            "blocked": False,
            "triggered_defense": None,
            "decision": decision,
            "harm_label": harm_label,
        }

# TESTING

# if __name__ == "__main__":
#     print(os.getenv("NGROK_ATTACK_ENDPOINT"))
#     llm = LLM(model_name="qwen2.5:3b-instruct", model_type="ollama", base_url=os.getenv("NGROK_SYSTEM_ENDPOINT"))
#     response = llm.generate_response("What is the capital of France?")
#     print(response)