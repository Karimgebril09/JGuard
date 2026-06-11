from datetime import datetime, timezone
import importlib
from typing import Any, TypedDict

from backend.app.models.chat_models import ChatRequest
from defenses.obfuscation.pipeline import run_obfuscation_guard

_CHAT_HISTORY: list[dict[str, str]] = []
_AGENT_CHAT_HISTORY: list[dict[str, str]] = []
_MAS_APP: Any | None = None


class ChatResult(TypedDict):
    reply: str
    blocked: bool
    triggered_defense: str | None
    decision: str | None
    harm_label: str | None
    timestamp: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_obfuscation_if_enabled(payload: ChatRequest) -> tuple[str, str | None, str | None, bool]:
    clean_prompt = payload.prompt
    decision: str | None = None
    harm_label: str | None = None
    blocked = False

    if payload.obfuscation_protection:
        guard_result = run_obfuscation_guard(payload.prompt)
        clean_prompt = str(guard_result.get("clean_text", payload.prompt))
        decision = str(guard_result.get("decision")) if guard_result.get("decision") is not None else None
        harm_label = str(guard_result.get("harm_label")) if guard_result.get("harm_label") is not None else None
        blocked = not bool(guard_result.get("is_safe", True))

    return clean_prompt, decision, harm_label, blocked


def run_chat(payload: ChatRequest) -> ChatResult:
    _CHAT_HISTORY.extend([message.model_dump() for message in payload.history])
    _CHAT_HISTORY.append({"role": "user", "content": payload.prompt})

    clean_prompt, decision, harm_label, blocked = _apply_obfuscation_if_enabled(payload)
    if blocked:
        blocked_reply = (
            "Request blocked by obfuscation guard. "
            f"decision={decision or 'unknown'}, harm_label={harm_label or 'unknown'}"
        )
        _CHAT_HISTORY.append({"role": "assistant", "content": blocked_reply})
        return {
            "reply": blocked_reply,
            "blocked": True,
            "triggered_defense": "obfuscation",
            "decision": decision,
            "harm_label": harm_label,
            "timestamp": _now_iso(),
        }

    reply = _dispatch_llm_reply(payload=payload, prompt_text=clean_prompt)

    _CHAT_HISTORY.append({"role": "assistant", "content": reply})
    return {
        "reply": reply,
        "blocked": False,
        "triggered_defense": None,
        "decision": decision,
        "harm_label": harm_label,
        "timestamp": _now_iso(),
    }


def _get_mas_app() -> Any:
    global _MAS_APP
    if _MAS_APP is not None:
        return _MAS_APP

    try:
        from system.multi_agentic.agents import app as mas_app_module
    except Exception as exc:
        raise RuntimeError(f"Failed to import MAS modules: {exc}") from exc

    # Compile a fresh runtime graph here so backend can invoke MAS directly.
    # This path intentionally avoids custom checkpointer coupling for API runtime stability.
    _MAS_APP = mas_app_module.graph.compile()
    return _MAS_APP


def run_agent_chat(payload: ChatRequest) -> ChatResult:
    _AGENT_CHAT_HISTORY.extend([message.model_dump() for message in payload.history])
    _AGENT_CHAT_HISTORY.append({"role": "user", "content": payload.prompt})

    clean_prompt, decision, harm_label, blocked = _apply_obfuscation_if_enabled(payload)
    if blocked:
        blocked_reply = (
            "Request blocked by obfuscation guard. "
            f"decision={decision or 'unknown'}, harm_label={harm_label or 'unknown'}"
        )
        _AGENT_CHAT_HISTORY.append({"role": "assistant", "content": blocked_reply})
        return {
            "reply": blocked_reply,
            "blocked": True,
            "triggered_defense": "obfuscation",
            "decision": decision,
            "harm_label": harm_label,
            "timestamp": _now_iso(),
        }

    app = _get_mas_app()
    try:
        final_state = app.invoke(
            {
                "user_message": clean_prompt,
                "messages": [],
            },
            config={"recursion_limit": 30},
        )
    except Exception as exc:
        raise RuntimeError(f"Agent-based MAS call failed: {exc}") from exc

    reply = _extract_mas_reply(final_state)
    _AGENT_CHAT_HISTORY.append({"role": "assistant", "content": reply})

    return {
        "reply": reply,
        "blocked": False,
        "triggered_defense": None,
        "decision": decision,
        "harm_label": harm_label,
        "timestamp": _now_iso(),
    }


def _extract_mas_reply(final_state: dict[str, Any]) -> str:
    response = final_state.get("response")
    if isinstance(response, str) and response.strip():
        return response

    messages = final_state.get("messages", [])
    for message in reversed(messages):
        msg_type = getattr(message, "type", "")
        content = getattr(message, "content", "")
        if msg_type == "ai":
            if isinstance(content, str) and content.strip():
                return content
            return str(content)

    return "Agent system returned no response."


def _dispatch_llm_reply(payload: ChatRequest, prompt_text: str) -> str:
    if payload.local_llm:
        return _call_local_ollama(payload, prompt_text)

    provider = _resolve_cloud_provider(payload.llm_type)
    if provider == "gemini":
        return _call_gemini(payload, prompt_text)
    return _call_openai(payload, prompt_text)


def _resolve_cloud_provider(llm_type: str) -> str:
    lowered = llm_type.strip().lower()
    if lowered.startswith("gemini") or lowered.startswith("google/"):
        return "gemini"
    return "openai"


def _call_local_ollama(payload: ChatRequest, prompt_text: str) -> str:
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
    messages.append({"role": "user", "content": prompt_text})

    try:
        response: dict[str, Any] = ollama.chat(model=payload.llm_type, messages=messages)
        message = response.get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        return "Local model returned an empty response."
    except Exception as exc:
        raise RuntimeError(f"Local Ollama call failed: {exc}") from exc


def _call_gemini(payload: ChatRequest, prompt_text: str) -> str:
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
            f"user: {prompt_text}\n"
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


def _call_openai(payload: ChatRequest, prompt_text: str) -> str:
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
        messages.append({"role": "user", "content": prompt_text})

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
    _AGENT_CHAT_HISTORY.clear()
    return {"success": True}
