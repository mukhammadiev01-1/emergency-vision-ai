"""API Configuration Module."""
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """API configuration settings loaded from environment variables."""

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = False
    API_TITLE: str = "Emergency Vision AI API"
    API_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"
    API_CORS_ORIGINS: List[str] = ["*"]

    # Model / Worker integration mode (e.g. "direct", "redis", "mock")
    WORKER_MODE: str = "direct"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = APISettings()
