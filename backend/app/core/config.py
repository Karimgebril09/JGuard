from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True)
class Settings:
    app_name: str = "JGuard Backend"
    app_version: str = "0.1.0"
    app_description: str = "FastAPI backend scaffold for the JGuard WinUI dashboard."
    environment: str = "development"
    api_prefix: str = "/api"
    allowed_origins: list[str] | None = None


def _parse_allowed_origins() -> list[str]:
    raw_value = os.getenv("JGUARD_ALLOWED_ORIGINS", "*")
    origins = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
    return origins or ["*"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        environment=os.getenv("JGUARD_ENV", "development"),
        api_prefix=os.getenv("JGUARD_API_PREFIX", "/api"),
        allowed_origins=_parse_allowed_origins(),
    )