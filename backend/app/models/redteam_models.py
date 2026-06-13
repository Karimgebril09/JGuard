from typing import Literal

from pydantic import BaseModel


StrategyType = Literal["tool_based", "custom_atj"]
ToolFrameworkType = Literal["promptfoo", "garak", "deepteam"]
CampaignStatusType = Literal["started", "running", "completed", "failed", "stopped"]


class ModelRef(BaseModel):
    provider: str
    model_name: str


class LaunchCampaignRequest(BaseModel):
    strategy: StrategyType
    tool_framework: ToolFrameworkType | None = None
    # attacker_model: ModelRef | None = None
    # target_model: ModelRef | None = None
    # judge_model: ModelRef | None = None
    # obfuscation_layer: bool 
    # obfuscation_intensity: int
    # multi_turn_layer: bool
    # multi_turn_count: int
    # roleplaying_layer: bool
    # roleplaying_persona: str


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
