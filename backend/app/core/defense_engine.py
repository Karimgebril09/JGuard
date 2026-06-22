from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request

from backend.app.models.chat_models import SessionConfig
from system.single_llm.llm import LLM
from system.multi_agentic.mas_guard import MASGuard

llm_system_prompt = """
You are a helpful, accurate, and concise AI assistant.

Always provide clear and useful answers to the user's questions.

Important formatting rules:

* Return plain text only.
* Do NOT use Markdown formatting.
* Do NOT use code fences (```).
* Do NOT include emojis or special characters like asterisks.
* Do NOT use headings, bullet points, numbered lists, tables, or Markdown symbols such as *, #, _, >, or ` unless explicitly requested by the user.
* When showing code, output the code as plain text without Markdown code blocks.
* Keep responses well-structured using plain text and line breaks only.

If a request is ambiguous, ask a clarifying question before answering.
"""


@dataclass
class SessionState:
    session_id: str
    history: list[dict[str, str]]
    llm: Optional[LLM]
    config: SessionConfig
    created_at: datetime
    last_active: datetime
    meta: dict[str, Any] = field(default_factory=dict)
    mas_guard: Optional[MASGuard] = None
    mas_app: Any | None = None


class SessionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, SessionState] = {}

    def create_session(self, config: SessionConfig) -> SessionState:
        now = datetime.now(timezone.utc)

        llm: Optional[LLM] = None
        mas_guard: Optional[MASGuard] = None
        mas_app: Any = None

        if config.chat_mode == "foundational":
            assert config.llm_type, "llm_type is required for foundational mode"
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
                base_url=config.llm_base_url,
                system_prompt=llm_system_prompt.strip(),
            )
        else:
            mas_guard = MASGuard(
                obfuscation_protection=config.obfuscation_protection,
                roleplay_protection=config.roleplay_protection,
                pii_protection=config.pii_protection,
                pii_strategy=config.pii_strategy,
                multi_turn_protection=config.multi_turn_protection,
            )
            mas_app = _build_mas_app(
                web_search_protection=config.web_search_protection,
                code_execution_protection=config.code_execution_protection,
                code_deep_check=config.code_deep_check,
                rag_protection=config.rag_protection,
                email_protection=config.email_protection,
                document_protection=config.document_protection,
            )

        session = SessionState(
            session_id=str(uuid4()),
            history=[],
            llm=llm,
            config=config,
            created_at=now,
            last_active=now,
            mas_guard=mas_guard,
            mas_app=mas_app,
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

    def list_sessions(self) -> list[SessionState]:
        with self._lock:
            return list(self._sessions.values())

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


def get_session_store(request: Request) -> SessionStore:
    session_store = getattr(request.app.state, "session_store", None)
    if session_store is None:
        raise RuntimeError("Session store is not initialized.")
    return session_store


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_session_chat(session: SessionState, prompt: str) -> dict[str, Any]:
    session.history.append({"role": "user", "content": prompt})

    if session.config.chat_mode == "agent":
        result = _run_agent_chat(session, prompt)
    else:
        if session.llm is None:
            raise RuntimeError("LLM is not initialized for this session.")
        result = session.llm.chat_secure(prompt=prompt, history=session.history)

    session.history.append({"role": "assistant", "content": str(result["reply"])})
    result["timestamp"] = _now_iso()
    return result


def _run_agent_chat(session: SessionState, prompt: str) -> dict[str, Any]:
    if session.mas_guard is None or session.mas_app is None:
        raise RuntimeError("MAS app is not initialized for this session.")

    guard_result = session.mas_guard.secure_input(prompt)
    if guard_result["blocked"]:
        return guard_result

    clean_prompt: str = guard_result["clean_prompt"]
    print(f"MAS GUARD PASSED. Cleaned prompt: {clean_prompt}")

    try:
        final_state = session.mas_app.invoke(
            {
                "user_message": clean_prompt,
                "messages": [],
            },
            config={
                "recursion_limit": 30,
                "configurable": {
                    "mas_guard": session.mas_guard,
                },
            },
        )
    except Exception as exc:
        raise RuntimeError(f"Agent-based MAS call failed: {exc}") from exc

    if "response" in final_state:
        session.meta["agent_response"] = final_state.get("response")
    if "next_action" in final_state:
        session.meta["agent_next_action"] = final_state.get("next_action")

    return {
        "reply": _extract_mas_reply(final_state),
        "blocked": False,
        "triggered_defenses": guard_result["triggered_defenses"],
        "harm_label": guard_result["harm_label"],
        "clean_prompt": clean_prompt,
    }


def _build_mas_app(
    web_search_protection: bool = True,
    code_execution_protection: bool = True,
    code_deep_check: bool = True,
    rag_protection: bool = True,
    email_protection: bool = True,
    document_protection: bool = True,
) -> Any:
    try:
        from system.multi_agentic.agents.app import build_mas_app
    except Exception as exc:
        raise RuntimeError(f"Failed to import MAS modules: {exc}") from exc

    return build_mas_app(
        web_search_protection=web_search_protection,
        code_execution_protection=code_execution_protection,
        code_deep_check=code_deep_check,
        rag_protection=rag_protection,
        email_protection=email_protection,
        document_protection=document_protection,
    )


def _extract_mas_reply(final_state: dict[str, Any]) -> str:
    response = final_state.get("response")
    if isinstance(response, str) and response.strip():
        return response

    for message in reversed(final_state.get("messages", [])):
        if getattr(message, "type", "") == "ai":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) and content.strip() else str(content)

    return "Agent system returned no response."


def _resolve_cloud_provider(llm_type: str) -> str:
    lowered = llm_type.strip().lower()
    if lowered.startswith("gemini") or lowered.startswith("google/"):
        return "gemini"
    return "openai"
