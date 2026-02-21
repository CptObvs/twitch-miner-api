"""
Application configuration using Pydantic Settings.
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    DOCKER_PORT_BASE: int = 5000
    DOCKER_STOP_TIMEOUT: int = 30

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

    # Startup behavior
    RUN_MIGRATIONS_ON_STARTUP: bool = True

    # Instance limits
    MAX_INSTANCES_PER_USER: int = 2  # Maximum instances for "user" role

    # Subprocess miner
    MINER_REPO_PATH: str | None = None
    LOG_HISTORY_LINES: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
