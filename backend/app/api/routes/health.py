from fastapi import APIRouter

from backend.app.core.config import get_settings
from backend.app.models.health_models import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/ping")
def ping() -> dict[str, str]:
    return {"message": "server is up and running"}


@router.get("", response_model=HealthResponse)
def get_health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )