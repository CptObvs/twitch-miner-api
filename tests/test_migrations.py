"""Exercise upgrade and downgrade against a private SQLite database."""

import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config


def test_ip_retention_migration_keeps_recent_data_and_downgrades(tmp_path):
    path = tmp_path / "migration.db"
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path.as_posix()}")
    command.upgrade(config, "0005_connected_ips")
    with sqlite3.connect(path) as db:
        db.executemany(
            "INSERT INTO connected_ips (ip_address, first_seen, last_seen, request_count) VALUES (?, ?, ?, ?)",
            [(f"ip-{i:04}", "2026-01-01T00:00:00", f"2026-01-{1 + i // 100:02}T00:{i % 60:02}:00", i)
             for i in range(1100)],
        )
    command.upgrade(config, "head")
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM connected_ips").fetchone()[0] == 1000
        assert db.execute("SELECT count(*) FROM connected_ips WHERE ip_address < 'ip-0100'").fetchone()[0] == 0
        assert db.execute("SELECT request_count FROM connected_ips WHERE ip_address='ip-1099'").fetchone()[0] == 1099
        assert db.execute("SELECT count(*) FROM connected_ips WHERE last_seen LIKE '%T%'").fetchone()[0] == 0
        assert db.execute("SELECT name FROM sqlite_master WHERE name='ix_connected_ips_last_seen_ip'").fetchone()
    command.downgrade(config, "0005_connected_ips")
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM connected_ips").fetchone()[0] == 1000


def test_startup_migrations_use_configured_database(tmp_path, monkeypatch):
    import main
    path = tmp_path / "configured.db"
    monkeypatch.setattr(main.settings, "DATABASE_URL", f"sqlite+aiosqlite:///{path.as_posix()}")
    main._run_migrations()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "0006_limit_ip_tracking"
