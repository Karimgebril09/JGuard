from threading import Lock
from typing import Any

from defenders.multi_turn.integrated.inference.multi_turn_defender import MultiTurnDefender
from defenders.obfuscation.pipeline import run_obfuscation
from defenders.pii_detection.src.pii_engine import PIIEngine
from defenders.pii_detection.src.strategies import (
    BlockStrategy,
    HashStrategy,
    MaskStrategy,
    PartialMasking,
    PIIStrategy,
)
from defenders.role_playing.pipeline import run_role_playing_guard

_AGENT_NAMES = ("orchestrator", "coder", "research", "document", "rag", "email")

_default_blocked_reply = "I cannot fulfill that request."


def _resolve_pii_strategy(strategy_name: str) -> PIIStrategy:
    normalized = strategy_name.strip().lower()
    if normalized == "hash":
        return HashStrategy()
    if normalized == "block":
        return BlockStrategy()
    if normalized == "partial":
        return PartialMasking()
    return MaskStrategy()


class MASGuard:

    def __init__(
        self,
        obfuscation_protection: bool = False,
        roleplay_protection: bool = False,
        pii_protection: bool = False,
        pii_strategy: str = "mask",
        multi_turn_protection: bool = False,
    ) -> None:
        self.obfuscation_protection = obfuscation_protection
        self.roleplay_protection = roleplay_protection
        self.pii_protection = pii_protection
        self.pii_strategy = pii_strategy
        self.multi_turn_protection = multi_turn_protection

        self._pii_lock = Lock()
        self._pii_engine: PIIEngine | None = (
            PIIEngine(strategy=_resolve_pii_strategy(pii_strategy))
            if pii_protection
            else None
        )

        self._multi_turn_lock = Lock()
        if multi_turn_protection:
            self._defenders: dict[str, MultiTurnDefender] = {
                name: MultiTurnDefender() for name in _AGENT_NAMES
            }
            self._last_responses: dict[str, str] = {name: "" for name in _AGENT_NAMES}
        else:
            self._defenders = {}
            self._last_responses = {}

    def _apply_pii(self, text: str) -> tuple[str, bool]:
        if not self.pii_protection or self._pii_engine is None:
            return text, False
        print("PII ENGINE RUNNING")
        with self._pii_lock:
            result = str(self._pii_engine.process(text))
        if result == "<BLOCKED>":
            return result, True
        return result, False

    def _apply_obfuscation(self, text: str) -> tuple[str, str | None, str | None, bool]:
        if not self.obfuscation_protection:
            return text, None, None, False
        print("OBFUSCATION ENGINE RUNNING")
        result = run_obfuscation(text)
        clean_text = str(result.get("clean_text", text))
        decision = str(result.get("decision")) if result.get("decision") is not None else None
        harm_label = str(result.get("harm_label")) if result.get("harm_label") is not None else None
        blocked = not bool(result.get("is_safe", True))
        return clean_text, decision, harm_label, blocked

    def _apply_role_playing(self, text: str) -> bool:
        if not self.roleplay_protection:
            return False
        print("ROLE-PLAYING GUARD RUNNING")
        result = run_role_playing_guard(text)
        action = str(result.get("action", "")).strip().lower()
        if action == "block":
            return True
        return not bool(result.get("is_safe", True))

    def check_multi_turn(self, agent_name: str, prompt: str) -> bool:
        if not self.multi_turn_protection:
            return False
        print("CHECKING MULTI-TURN DEFENSE FOR AGENT: ", agent_name)
        with self._multi_turn_lock:
            defender = self._defenders[agent_name]
            last = self._last_responses.get(agent_name, "")
            prediction = defender.predict(prompt, last)
        return int(prediction) == 1

    def update_last_response(self, agent_name: str, response: str) -> None:
        if not self.multi_turn_protection:
            return
        print("UPDATING LAST RESPONSE FOR AGENT: ", agent_name)
        with self._multi_turn_lock:
            self._last_responses[agent_name] = response

    def secure_input(self, prompt: str) -> dict[str, Any]:
        """
        Run all defenses on the user prompt before it reaches the MAS.
        """
        triggered_defenses: list[str] = []
        harm_label: str | None = None

        print("MAS INPUT GUARD RUNNING")

        pii_prompt, pii_blocked = self._apply_pii(prompt)
        if pii_blocked:
            triggered_defenses.append("PII Detector")
            print("BLOCKED: PII detected in user prompt.")
            return {
                "reply": _default_blocked_reply + " Please do not include PII in your requests.",
                "blocked": True,
                "triggered_defenses": triggered_defenses,
                "harm_label": None,
                "clean_prompt": pii_prompt,
            }

        clean_prompt, _, harm_label, general_harm_detected = self._apply_obfuscation(pii_prompt)
        if general_harm_detected:
            triggered_defenses.append("General Harm Detector")

        # Re-check PII on the deobfuscated text to catch PII hidden inside encodings
        if self.obfuscation_protection and self.pii_protection:
            clean_prompt, pii_blocked = self._apply_pii(clean_prompt)
            if pii_blocked:
                triggered_defenses.append("PII Detector")
                print("BLOCKED: PII detected in user prompt after deobfuscation.")
                return {
                    "reply": _default_blocked_reply + " Please do not include PII in your requests.",
                    "blocked": True,
                    "triggered_defenses": triggered_defenses,
                    "harm_label": None,
                    "clean_prompt": clean_prompt,
                }

        roleplay_blocked = self._apply_role_playing(clean_prompt)
        if roleplay_blocked:
            triggered_defenses.append("Role-Playing Defender")

        if triggered_defenses:
            print(f"BLOCKED: Defenses triggered - {', '.join(triggered_defenses)}. Harm label: {harm_label}")
            return {
                "reply": _default_blocked_reply,
                "blocked": True,
                "triggered_defenses": triggered_defenses,
                "harm_label": harm_label,
                "clean_prompt": clean_prompt,
            }

        return {
            "reply": None,
            "blocked": False,
            "triggered_defenses": [],
            "harm_label": harm_label,
            "clean_prompt": clean_prompt,
        }

    def check_message(self, text: str) -> tuple[str, bool, list[str]]:
        triggered: list[str] = []

        print("AI GATEWAY RUNNING")

        pii_text, pii_blocked = self._apply_pii(text)
        if pii_blocked:
            triggered.append("PII Detector")
            return _default_blocked_reply, True, triggered

        clean_text, _, _, obf_blocked = self._apply_obfuscation(pii_text)
        if obf_blocked:
            triggered.append("General Harm Detector")

        if self.obfuscation_protection and self.pii_protection:
            clean_text, pii_blocked = self._apply_pii(clean_text)
            if pii_blocked:
                triggered.append("PII Detector")
                return _default_blocked_reply, True, triggered

        rp_blocked = self._apply_role_playing(clean_text)
        if rp_blocked:
            triggered.append("Role-Playing Defender")

        if triggered:
            print(f"AI GATEWAY HAS BLOCKED: Defenses triggered - {', '.join(triggered)}.")
            return _default_blocked_reply, True, triggered

        return clean_text, False, []
