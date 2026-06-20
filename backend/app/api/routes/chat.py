from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core import defense_engine
from backend.app.models.chat_models import (
    ChatHistoryMessage,
    ChatResponse,
    SessionChatRequest,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionDeleteResponse,
    SessionHistoryResponse,
    SessionListResponse,
    SessionSummary,
)


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=SessionListResponse)
def list_sessions(
    session_store: defense_engine.SessionStore = Depends(defense_engine.get_session_store),
) -> SessionListResponse:
    sessions = session_store.list_sessions()
    return SessionListResponse(
        sessions=[
            SessionSummary(
                session_id=s.session_id,
                created_at=s.created_at.isoformat(),
                last_active=s.last_active.isoformat(),
                config=s.config,
                meta=s.meta,
            )
            for s in sessions
        ]
    )


@router.post("", response_model=SessionCreateResponse)
def create_session(
    payload: SessionCreateRequest,
    session_store: defense_engine.SessionStore = Depends(defense_engine.get_session_store),
) -> SessionCreateResponse:
    session = session_store.create_session(payload.config)
    return SessionCreateResponse(
        session_id=session.session_id,
        created_at=session.created_at.isoformat(),
        last_active=session.last_active.isoformat(),
        config=session.config,
    )


@router.delete("/{session_id}", response_model=SessionDeleteResponse)
def delete_session(
    session_id: str,
    session_store: defense_engine.SessionStore = Depends(defense_engine.get_session_store),
) -> SessionDeleteResponse:
    deleted = session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return SessionDeleteResponse(success=True, session_id=session_id)


@router.post("/{session_id}/chat", response_model=ChatResponse)
def session_chat(
    session_id: str,
    payload: SessionChatRequest,
    session_store: defense_engine.SessionStore = Depends(defense_engine.get_session_store),
) -> ChatResponse:
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    try:
        result = defense_engine.run_session_chat(session=session, prompt=payload.prompt)
        return ChatResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
def session_history(
    session_id: str,
    session_store: defense_engine.SessionStore = Depends(defense_engine.get_session_store),
) -> SessionHistoryResponse:
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    history = [
        ChatHistoryMessage(role=str(message.get("role", "user")), content=str(message.get("content", "")))
        for message in session.history
    ]
    return SessionHistoryResponse(
        session_id=session.session_id,
        history=history,
        created_at=session.created_at.isoformat(),
        last_active=session.last_active.isoformat(),
        meta=session.meta,
    )
