from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ChatHistoryMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class SessionConfig(BaseModel):
    chat_mode: Literal["foundational", "agent"] = "foundational"
    local_llm: bool
    llm_api_key: str = ""
    llm_type: str = Field(..., min_length=1)
    llm_base_url: str | None = None
    obfuscation_protection: bool
    multi_turn_protection: bool
    roleplay_protection: bool
    pii_protection: bool
    pii_strategy: str = "mask"

    @model_validator(mode="after")
    def validate_closed_source_fields(self) -> "SessionConfig":
        if not self.local_llm and not self.llm_api_key.strip():
            raise ValueError("llm_api_key is required when local_llm is false.")
        return self


class SessionCreateRequest(BaseModel):
    config: SessionConfig


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: str
    last_active: str
    config: SessionConfig


class SessionChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class SessionDeleteResponse(BaseModel):
    success: bool
    session_id: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: list[ChatHistoryMessage]
    created_at: str
    last_active: str
    meta: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    blocked: bool
    triggered_defense: str | None
    decision: str | None = None
    harm_label: str | None = None
    timestamp: str
