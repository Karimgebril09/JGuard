from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import Request

from backend.app.models.chat_models import ChatResponse, SessionConfig
from defenders.obfuscation.pipeline import run_obfuscation
from defenders.pii_detection.src.pii_engine import PIIEngine
from defenders.pii_detection.src.strategies import BlockStrategy, EncryptStrategy, MaskStrategy, PIIStrategy


@dataclass
class RuntimeResources:
    pii_engine: PIIEngine
    mas_app: Any | None
    pii_lock: Lock
    mas_lock: Lock
    obfuscation_lock: Lock
    obfuscation_warmed_up: bool = False


@dataclass
class SessionState:
    session_id: str
    history: list[dict[str, str]]
    multi_turn_state: dict[str, Any]
    config: SessionConfig
    created_at: datetime
    last_active: datetime
    meta: dict[str, Any] = field(default_factory=dict)


class SessionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, SessionState] = {}

    def create_session(self, config: SessionConfig) -> SessionState:
        now = datetime.now(timezone.utc)
        session = SessionState(
            session_id=str(uuid4()),
            history=[],
            multi_turn_state={},
            config=config,
            created_at=now,
            last_active=now,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> SessionState | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_active = datetime.now(timezone.utc)
            return session

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


def get_runtime_resources(request: Request) -> RuntimeResources:
    runtime = getattr(request.app.state, "defense_runtime", None)
    if runtime is None:
        raise RuntimeError("Runtime resources are not initialized.")
    return runtime


def get_session_store(request: Request) -> SessionStore:
    session_store = getattr(request.app.state, "session_store", None)
    if session_store is None:
        raise RuntimeError("Session store is not initialized.")
    return session_store


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_runtime_resources() -> RuntimeResources:
    pii_engine = PIIEngine(strategy=MaskStrategy())
    runtime = RuntimeResources(
        pii_engine=pii_engine,
        mas_app=None,
        pii_lock=Lock(),
        mas_lock=Lock(),
        obfuscation_lock=Lock(),
        obfuscation_warmed_up=False,
    )
    _warmup_obfuscation(runtime)
    return runtime


def shutdown_runtime_resources() -> None:
    return


def _warmup_obfuscation(runtime: RuntimeResources) -> bool:
    with runtime.obfuscation_lock:
        if runtime.obfuscation_warmed_up:
            return True
        run_obfuscation("warmup")
        runtime.obfuscation_warmed_up = True
    return True


def _apply_obfuscation_if_enabled(
    session: SessionState,
    prompt: str,
    runtime: RuntimeResources,
) -> tuple[str, str | None, str | None, bool]:
    if not session.config.obfuscation_protection:
        return prompt, None, None, False

    _warmup_obfuscation(runtime)
    
    clean_prompt = prompt
    decision: str | None = None
    harm_label: str | None = None
    blocked = False

    result = run_obfuscation(prompt)
    clean_prompt = str(result.get("clean_text", prompt))
    decision = str(result.get("decision")) if result.get("decision") is not None else None
    harm_label = str(result.get("harm_label")) if result.get("harm_label") is not None else None
    blocked = not bool(result.get("is_safe", True))

    return clean_prompt, decision, harm_label, blocked


def _resolve_pii_strategy(pii_strategy: str) -> PIIStrategy:
    strategy_name = pii_strategy.strip().lower()
    if strategy_name == "encrypt":
        return EncryptStrategy()
    if strategy_name == "block":
        return BlockStrategy()
    return MaskStrategy()  # default to masking


def _get_pii_engine(runtime: RuntimeResources) -> PIIEngine:
    return runtime.pii_engine


def _apply_pii_if_enabled(session: SessionState, prompt_text: str, runtime: RuntimeResources) -> tuple[str, bool]:
    if not session.config.pii_protection:
        return prompt_text, False

    pii_engine = _get_pii_engine(runtime)
    with runtime.pii_lock:
        pii_engine.set_strategy(_resolve_pii_strategy(session.config.pii_strategy))
        pii_result = str(pii_engine.process(prompt_text))

    if pii_result == "[BLOCKED: PII DETECTED]":
        return pii_result, True

    return pii_result, False


def run_session_chat(session: SessionState, prompt: str, runtime: RuntimeResources) -> dict[str, str | bool | None]:
    session.history.append({"role": "user", "content": prompt})

    clean_prompt, decision, harm_label, blocked = _apply_obfuscation_if_enabled(session, prompt, runtime)
    if blocked:
        blocked_reply = (
            "Request blocked by obfuscation guard. "
            f"decision={decision or 'unknown'}, harm_label={harm_label or 'unknown'}"
        )
        session.history.append({"role": "assistant", "content": blocked_reply})
        return {
            "reply": blocked_reply,
            "blocked": True,
            "triggered_defense": "obfuscation",
            "decision": decision,
            "harm_label": harm_label,
            "timestamp": _now_iso(),
        }

    pii_prompt, pii_blocked = _apply_pii_if_enabled(session=session, prompt_text=clean_prompt, runtime=runtime)
    if pii_blocked:
        blocked_reply = "Request blocked by pii model."
        session.history.append({"role": "assistant", "content": blocked_reply})
        return {
            "reply": blocked_reply,
            "blocked": True,
            "triggered_defense": "pii",
            "decision": decision,
            "harm_label": harm_label,
            "timestamp": _now_iso(),
        }

    if session.config.chat_mode == "agent":
        reply = _dispatch_agent_reply(session=session, prompt_text=pii_prompt, runtime=runtime)
    else:
        reply = _dispatch_foundational_reply(session=session, prompt_text=pii_prompt)

    session.history.append({"role": "assistant", "content": reply})
    return {
        "reply": reply,
        "blocked": False,
        "triggered_defense": None,
        "decision": decision,
        "harm_label": harm_label,
        "timestamp": _now_iso(),
    }


def _build_mas_app() -> Any:
    try:
        from system.multi_agentic.agents import app as mas_app_module
    except Exception as exc:
        raise RuntimeError(f"Failed to import MAS modules: {exc}") from exc

    return mas_app_module.graph.compile()


def _dispatch_agent_reply(session: SessionState, prompt_text: str, runtime: RuntimeResources) -> str:
    with runtime.mas_lock:
        if runtime.mas_app is None:
            runtime.mas_app = _build_mas_app()
        mas_app: Any = runtime.mas_app
    
    prior_messages = session.multi_turn_state.get("messages", [])
    try:
        final_state = mas_app.invoke(
            {
                "user_message": prompt_text,
                "messages": prior_messages,
            },
            config={"recursion_limit": 30},
        )
    except Exception as exc:
        raise RuntimeError(f"Agent-based MAS call failed: {exc}") from exc

    session.multi_turn_state["messages"] = final_state.get("messages", prior_messages)
    if "response" in final_state:
        session.multi_turn_state["response"] = final_state.get("response")
    if "next_action" in final_state:
        session.multi_turn_state["next_action"] = final_state.get("next_action")

    return _extract_mas_reply(final_state)


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


def _dispatch_foundational_reply(session: SessionState, prompt_text: str) -> str:
    if session.config.local_llm:
        return _call_local_ollama(session, prompt_text)

    provider = _resolve_cloud_provider(session.config.llm_type)
    if provider == "gemini":
        return _call_gemini(session, prompt_text)
    return _call_openai(session, prompt_text)


def _resolve_cloud_provider(llm_type: str) -> str:
    lowered = llm_type.strip().lower()
    if lowered.startswith("gemini") or lowered.startswith("google/"):
        return "gemini"
    return "openai"


def _session_messages_for_llm(session: SessionState, prompt_text: str) -> list[dict[str, str]]:
    prior_messages = session.history[:-1]
    messages = [
        {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }
        for message in prior_messages
    ]
    messages.append({"role": "user", "content": prompt_text})
    return messages


def _call_local_ollama(session: SessionState, prompt_text: str) -> str:
    try:
        ollama = importlib.import_module("ollama")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ollama package is not installed. Install backend requirements first.") from exc

    messages = _session_messages_for_llm(session, prompt_text)

    try:
        response: dict[str, Any] = ollama.chat(model=session.config.llm_type, messages=messages)
        message = response.get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        return "Local model returned an empty response."
    except Exception as exc:
        raise RuntimeError(f"Local Ollama call failed: {exc}") from exc


def _call_gemini(session: SessionState, prompt_text: str) -> str:
    try:
        genai = importlib.import_module("google.genai")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Gemini package is not installed. Install backend requirements first."
        ) from exc

    try:
        client = genai.Client(api_key=session.config.llm_api_key)

        history_lines = [
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in session.history[:-1]
        ]
        history_text = "\n".join(history_lines)
        full_prompt = (
            "You are in a chat session. Continue the assistant reply based on the conversation.\n\n"
            f"Conversation:\n{history_text}\n"
            f"user: {prompt_text}\n"
            "assistant:"
        )

        response = client.models.generate_content(
            model=session.config.llm_type,
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


def _call_openai(session: SessionState, prompt_text: str) -> str:
    try:
        openai_module = importlib.import_module("openai")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenAI package is not installed. Install backend requirements first."
        ) from exc

    try:
        client = openai_module.OpenAI(api_key=session.config.llm_api_key)
        messages = _session_messages_for_llm(session, prompt_text)

        response = client.chat.completions.create(
            model=session.config.llm_type,
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
