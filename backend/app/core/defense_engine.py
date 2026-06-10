from datetime import datetime, timezone
import importlib
from typing import Any

from backend.app.models.chat_models import ChatRequest
from backend.app.models.system_models import SystemConfigRequest


_SYSTEM_STATE: dict[str, object] = {
    "architecture": "foundational_llm",
    "obfuscation_active": True,
    "multi_turn_active": True,
    "roleplay_active": True,
}

_CHAT_HISTORY: list[dict[str, str]] = []


def get_system_status() -> dict[str, object]:
    shields_enabled = bool(
        _SYSTEM_STATE["obfuscation_active"]
        or _SYSTEM_STATE["multi_turn_active"]
        or _SYSTEM_STATE["roleplay_active"]
    )
    return {
        "architecture": _SYSTEM_STATE["architecture"],
        "shields_enabled": shields_enabled,
        "obfuscation_active": _SYSTEM_STATE["obfuscation_active"],
        "multi_turn_active": _SYSTEM_STATE["multi_turn_active"],
        "roleplay_active": _SYSTEM_STATE["roleplay_active"],
    }


def update_system_config(payload: SystemConfigRequest) -> dict[str, object]:
    _SYSTEM_STATE["architecture"] = payload.architecture
    _SYSTEM_STATE["obfuscation_active"] = payload.obfuscation_protection
    _SYSTEM_STATE["multi_turn_active"] = payload.multi_turn_protection
    _SYSTEM_STATE["roleplay_active"] = payload.roleplay_protection

    shield_status = "enabled" if any(
        [
            payload.obfuscation_protection,
            payload.multi_turn_protection,
            payload.roleplay_protection,
        ]
    ) else "disabled"
    return {"success": True, "shield_status": shield_status}


def run_chat(mode: str, payload: ChatRequest) -> dict[str, object]:
    _CHAT_HISTORY.extend([message.model_dump() for message in payload.history])
    _CHAT_HISTORY.append({"role": "user", "content": payload.prompt})

    triggered_defense = None
    blocked = False
    lowered_prompt = payload.prompt.lower()
    if payload.roleplay_protection and "pretend" in lowered_prompt:
        triggered_defense = "roleplay"
        blocked = True
    elif payload.obfuscation_protection and any(ch.isdigit() for ch in payload.prompt):
        triggered_defense = "obfuscation"
    elif payload.multi_turn_protection and len(payload.history) >= 4:
        triggered_defense = "multi_turn"

    reply = (
        "Request blocked by safety defense layer."
        if blocked
        else _dispatch_llm_reply(mode=mode, payload=payload)
    )

    _CHAT_HISTORY.append({"role": "assistant", "content": reply})
    return {
        "reply": reply,
        "blocked": blocked,
        "triggered_defense": triggered_defense,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _dispatch_llm_reply(mode: str, payload: ChatRequest) -> str:
    if payload.local_llm:
        return _call_local_ollama(payload)

    # Placeholder until closed-source integration is wired.
    return f"[{mode}] closed-source provider path is scaffolded but not implemented yet."


def _call_local_ollama(payload: ChatRequest) -> str:
    try:
        ollama = importlib.import_module("ollama")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ollama package is not installed. Install backend requirements first.") from exc

    messages: list[dict[str, str]] = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in payload.history
    ]
    messages.append({"role": "user", "content": payload.prompt})

    try:
        response: dict[str, Any] = ollama.chat(model=payload.llm_type, messages=messages)
        message = response.get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        return "Local model returned an empty response."
    except Exception as exc:
        raise RuntimeError(f"Local Ollama call failed: {exc}") from exc


def clear_chat_history() -> dict[str, bool]:
    _CHAT_HISTORY.clear()
    return {"success": True}
