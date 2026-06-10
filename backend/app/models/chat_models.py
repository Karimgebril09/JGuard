from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatHistoryMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    local_llm: bool
    llm_api_key: str = ""
    llm_type: str = Field(..., min_length=1)
    obfuscation_protection: bool
    multi_turn_protection: bool
    roleplay_protection: bool
    history: list[ChatHistoryMessage] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_closed_source_fields(self) -> "ChatRequest":
        if not self.local_llm and not self.llm_api_key.strip():
            raise ValueError("llm_api_key is required when local_llm is false.")
        return self


class ChatResponse(BaseModel):
    reply: str
    blocked: bool
    triggered_defense: str | None
    timestamp: str


class ClearChatResponse(BaseModel):
    success: bool
