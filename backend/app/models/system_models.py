from typing import Literal

from pydantic import BaseModel


ArchitectureType = Literal["foundational_llm", "agent_based"]


class SystemStatusResponse(BaseModel):
    architecture: ArchitectureType
    shields_enabled: bool
    obfuscation_active: bool
    multi_turn_active: bool
    roleplay_active: bool


class SystemConfigRequest(BaseModel):
    architecture: ArchitectureType
    obfuscation_protection: bool
    multi_turn_protection: bool
    roleplay_protection: bool


class SystemConfigResponse(BaseModel):
    success: bool
    shield_status: str
