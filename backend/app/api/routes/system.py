from fastapi import APIRouter

from backend.app.core import defense_engine
from backend.app.models.system_models import (
    SystemConfigRequest,
    SystemConfigResponse,
    SystemStatusResponse,
)


router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status", response_model=SystemStatusResponse)
def get_status() -> SystemStatusResponse:
    return SystemStatusResponse.model_validate(defense_engine.get_system_status())


@router.post("/config", response_model=SystemConfigResponse)
def update_config(payload: SystemConfigRequest) -> SystemConfigResponse:
    result = defense_engine.update_system_config(payload)
    return SystemConfigResponse.model_validate(result)
