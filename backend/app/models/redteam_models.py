from typing import Literal

from pydantic import BaseModel


StrategyType = Literal["tool_based", "custom"]
ToolFrameworkType = Literal["promptfoo", "garak", "deepteam"]
CampaignStatusType = Literal["started", "running", "completed", "failed", "stopped"]


class ModelRef(BaseModel):
    provider: str
    model_name: str


class LaunchCampaignRequest(BaseModel):
    strategy: StrategyType
    tool_framework: ToolFrameworkType | None = None
    
    # custom
    custom_attack_type: str | None = None
    custom_harm_type: str | None = None
    num_samples: int | None = None

    judge_model: str | None = None
    attacker_model: str | None = None
    target_model: str | None = None

    judge_type: str | None = None
    attacker_type: str | None = None
    target_type: str | None = None

    judge_api_key: str | None = None
    attacker_api_key: str | None = None
    target_api_key: str | None = None

    judge_base_url: str | None = None
    attacker_base_url: str | None = None
    target_base_url: str | None = None

    obfuscation_protection: bool | None = None
    roleplay_protection: bool | None = None
    multi_turn_protection: bool | None = None
    pii_protection: bool | None = None
    pii_strategy: str | None = None


class LaunchCampaignResponse(BaseModel):
    campaign_id: str
    status: Literal["started"]
    timestamp: str


class CampaignStatusResponse(BaseModel):
    campaign_id: str
    status: CampaignStatusType
    log_lines: list[str]
    progress_pct: int


class StopCampaignResponse(BaseModel):
    success: bool
    campaign_id: str
    status: Literal["stopped"]
