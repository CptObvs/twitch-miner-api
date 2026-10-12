"""Run setup and startup migrations against the application's configured database."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

from app.core.config import settings


def run_migrations() -> None:
    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    url = make_url(settings.DATABASE_URL)
    if url.drivername == "sqlite+aiosqlite":
        url = url.set(drivername="sqlite")
    config.set_main_option(
        "sqlalchemy.url", url.render_as_string(hide_password=False).replace("%", "%%"),
    )
    command.upgrade(config, "head")
