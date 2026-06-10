from fastapi import APIRouter, HTTPException, status

from backend.app.core import redteam_runner
from backend.app.models.redteam_models import (
    CampaignStatusResponse,
    LaunchCampaignRequest,
    LaunchCampaignResponse,
    StopCampaignResponse,
)


router = APIRouter(prefix="/redteam", tags=["redteam"])


@router.post("/launch", response_model=LaunchCampaignResponse)
def launch(payload: LaunchCampaignRequest) -> LaunchCampaignResponse:
    return LaunchCampaignResponse(**redteam_runner.launch_campaign(payload))


@router.get("/status/{campaign_id}", response_model=CampaignStatusResponse)
def campaign_status(campaign_id: str) -> CampaignStatusResponse:
    try:
        return CampaignStatusResponse(**redteam_runner.get_campaign_status(campaign_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/stop/{campaign_id}", response_model=StopCampaignResponse)
def stop(campaign_id: str) -> StopCampaignResponse:
    try:
        return StopCampaignResponse(**redteam_runner.stop_campaign(campaign_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
