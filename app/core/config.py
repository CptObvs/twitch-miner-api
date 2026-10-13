"""
Application configuration using Pydantic Settings.
"""

from pathlib import Path
from typing import List
from urllib.parse import urlsplit

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, Field, field_validator


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = "Twitch Miner Backend"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENABLE_SWAGGER: bool = True
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"

    # Paths
    DATA_DIR: Path = Path(__file__).parent.parent.parent / "data"
    INSTANCES_DIR: Path = Path(__file__).parent.parent.parent / "data" / "instances"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///data/app.db"

    # Docker
    DOCKER_IMAGE: str = "rangermix/twitch-drops-miner:latest"
    V2_DOCKER_IMAGE: str = "rdavidoff/twitch-channel-points-miner-v2:latest"
    DOCKER_PULL_ON_START: bool = True
    DOCKER_PULL_TIMEOUT: int = 300
    DOCKER_TIMEZONE: str = "Europe/Berlin"
    DOCKER_PORT_BASE: int = 5000
    DOCKER_STOP_TIMEOUT: int = 30

    # Public browser origin when TLS terminates in front of the API.
    UI_PUBLIC_ORIGIN: str = ""

    # Security
    JWT_SECRET: str = "change-me-in-production-use-a-random-64-char-string"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 hours

    # CORS
    CORS_ORIGINS: List[str] = ["*"]  # Restrict in production!

    # Memory / scalability
    MEMORY_GC_ENABLED: bool = True
    MEMORY_GC_INTERVAL_SECONDS: int = 300
    MEMORY_GC_GENERATION: int = 2

    # Request logging
    API_REQUEST_LOGGING_ENABLED: bool = True

    # Automatic scanner bans (ordinary missing API resources are excluded)
    IP_BAN_HIT_THRESHOLD: int = Field(default=5, ge=1)
    IP_BAN_WINDOW_SECONDS: int = Field(default=60, ge=1)
    IP_BAN_DURATION_HOURS: int = Field(default=1, ge=1)

    # Instance limits
    MAX_INSTANCES_PER_USER: int = 2  # Maximum instances for "user" role

    # Migrations
    RUN_MIGRATIONS_ON_STARTUP: bool = True

    # Logs
    LOG_HISTORY_LINES: int = 200

    @field_validator("UI_PUBLIC_ORIGIN")
    @classmethod
    def validate_ui_public_origin(cls, value: str) -> str:
        if not value:
            return ""
        parts = urlsplit(value)
        if (any(char.isspace() for char in value) or parts.scheme not in {"http", "https"}
                or parts.path not in {"", "/"} or parts.query or parts.fragment
                or parts.username is not None or parts.password is not None):
            raise ValueError("UI_PUBLIC_ORIGIN must be an http(s) origin without a path, credentials, query or fragment")
        return str(AnyHttpUrl(value)).rstrip("/")

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
