from datetime import datetime, timezone
import importlib
from typing import Any

from backend.app.models.chat_models import ChatRequest

_CHAT_HISTORY: list[dict[str, str]] = []


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

    provider = _resolve_cloud_provider(payload.llm_type)
    if provider == "gemini":
        return _call_gemini(payload)
    return _call_openai(payload)


def _resolve_cloud_provider(llm_type: str) -> str:
    lowered = llm_type.strip().lower()
    if lowered.startswith("gemini") or lowered.startswith("google/"):
        return "gemini"
    return "openai"


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


def _call_gemini(payload: ChatRequest) -> str:
    try:
        genai = importlib.import_module("google.genai")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Gemini package is not installed. Install backend requirements first."
        ) from exc

    try:
        client = genai.Client(api_key=payload.llm_api_key)

        history_lines = [
            f"{message.role}: {message.content}"
            for message in payload.history
        ]
        history_text = "\n".join(history_lines)
        full_prompt = (
            "You are in a chat session. Continue the assistant reply based on the conversation.\n\n"
            f"Conversation:\n{history_text}\n"
            f"user: {payload.prompt}\n"
            "assistant:"
        )

        response = client.models.generate_content(
            model=payload.llm_type,
            contents=full_prompt,
        )
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        candidates = getattr(response, "candidates", None)
        if isinstance(candidates, list):
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None)
                if isinstance(parts, list):
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if isinstance(part_text, str) and part_text.strip():
                            return part_text

        return "Gemini returned an empty response."
    except Exception as exc:
        raise RuntimeError(f"Gemini call failed: {exc}") from exc


def _call_openai(payload: ChatRequest) -> str:
    try:
        openai_module = importlib.import_module("openai")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenAI package is not installed. Install backend requirements first."
        ) from exc

    try:
        client = openai_module.OpenAI(api_key=payload.llm_api_key)
        messages: list[dict[str, str]] = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in payload.history
        ]
        messages.append({"role": "user", "content": payload.prompt})

        response = client.chat.completions.create(
            model=payload.llm_type,
            messages=messages,
        )

        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            first = choices[0]
            message = getattr(first, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content

        return "OpenAI returned an empty response."
    except Exception as exc:
        raise RuntimeError(f"OpenAI call failed: {exc}") from exc


def clear_chat_history() -> dict[str, bool]:
    _CHAT_HISTORY.clear()
    return {"success": True}
