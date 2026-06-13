from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import Request

from backend.app.models.chat_models import SessionConfig
from system.single_llm.llm import LLM


@dataclass
class RuntimeResources:
    mas_app: Any | None
    mas_lock: Lock


@dataclass
class SessionState:
    session_id: str
    history: list[dict[str, str]]
    llm: LLM
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
        model_type = "ollama" if config.local_llm else _resolve_cloud_provider(config.llm_type)
        llm = LLM(
            model_name=config.llm_type,
            model_type=model_type,
            api_key=config.llm_api_key if not config.local_llm else None,
            obfuscation_protection=config.obfuscation_protection,
            roleplay_protection=config.roleplay_protection,
            multi_turn_protection=config.multi_turn_protection,
            pii_protection=config.pii_protection,
            pii_strategy=config.pii_strategy,
        )

        session = SessionState(
            session_id=str(uuid4()),
            history=[],
            llm=llm,
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
    return RuntimeResources(
        mas_app=None,
        mas_lock=Lock(),
    )


def shutdown_runtime_resources() -> None:
    return


def run_session_chat(session: SessionState, prompt: str, runtime: RuntimeResources) -> dict[str, str | bool | None]:
    session.history.append({"role": "user", "content": prompt})

    reply_fn = None
    if session.config.chat_mode == "agent":
        reply_fn = lambda clean_prompt: _dispatch_agent_reply(
            session=session,
            prompt_text=clean_prompt,
            runtime=runtime,
        )

    result = session.llm.chat_secure(
        prompt=prompt,
        history=session.history,
        reply_fn=reply_fn,
    )

    session.history.append({"role": "assistant", "content": str(result["reply"])})
    result["timestamp"] = _now_iso()
    return result


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

    prior_messages = session.meta.get("agent_messages", [])
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

    session.meta["agent_messages"] = final_state.get("messages", prior_messages)
    if "response" in final_state:
        session.meta["agent_response"] = final_state.get("response")
    if "next_action" in final_state:
        session.meta["agent_next_action"] = final_state.get("next_action")

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


def _resolve_cloud_provider(llm_type: str) -> str:
    lowered = llm_type.strip().lower()
    if lowered.startswith("gemini") or lowered.startswith("google/"):
        return "gemini"
    return "openai"
