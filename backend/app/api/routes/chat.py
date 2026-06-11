from fastapi import APIRouter, HTTPException, status

from backend.app.core import defense_engine
from backend.app.models.chat_models import ChatRequest, ChatResponse, ClearChatResponse


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/foundational", response_model=ChatResponse)
def foundational_chat(payload: ChatRequest) -> ChatResponse:
    try:
        return ChatResponse(**defense_engine.run_chat(payload=payload))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/agent", response_model=ChatResponse)
def agent_chat(payload: ChatRequest) -> ChatResponse:
    try:
        return ChatResponse(**defense_engine.run_agent_chat(payload=payload))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/clear", response_model=ClearChatResponse)
def clear_chat() -> ClearChatResponse:
    return ClearChatResponse(**defense_engine.clear_chat_history())
