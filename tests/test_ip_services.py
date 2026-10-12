"""Retention, recovery and scanner bans, using an isolated database."""

import asyncio
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from app.core.config import settings
from app.models.database import BannedIP, ConnectedIP
from app.services.ip_tracker_service import IPTrackerService
from app.services.ip_ban_service import IPBanService
from app.services import ip_ban_service as ban_module


def private_ip(index):
    return str(ip_address("10.0.0.1") + index)


async def test_tracker_retains_recent_1000_in_memory_and_database(db_session):
    tracker = IPTrackerService()
    for i in range(1000):
        tracker.record(private_ip(i))
    await tracker.flush(db_session)
    tracker.record(private_ip(0))  # Refresh the oldest IP before adding another.
    tracker.record(private_ip(1000))
    assert len(tracker._ips) == 1000
    assert private_ip(0) in tracker._ips
    assert private_ip(1) not in tracker._ips
    await tracker.flush(db_session)
    rows = await tracker.get_all(db_session)
    assert len(rows) == 1000
    assert rows[0]["ip_address"] == private_ip(1000)
    assert {row["ip_address"] for row in rows} == set(tracker._ips)
    assert await db_session.scalar(select(func.count()).select_from(ConnectedIP)) == 1000


async def test_tracker_load_prunes_old_database_and_retry_keeps_dirty_records(db_session, monkeypatch):
    start = datetime(2026, 1, 1)
    db_session.add_all([
        ConnectedIP(ip_address=private_ip(i), first_seen=start,
                    last_seen=start + timedelta(seconds=i), request_count=1)
        for i in range(1005)
    ])
    await db_session.commit()
    tracker = IPTrackerService()
    await tracker.load_from_db(db_session)
    assert len(tracker._ips) == 1000
    assert private_ip(0) not in tracker._ips
    assert next(iter(tracker._ips)) == private_ip(5)
    tracker.record(private_ip(1004))
    with monkeypatch.context() as patch:
        patch.setattr(db_session, "commit", AsyncMock(side_effect=RuntimeError("temporary failure")))
        with pytest.raises(RuntimeError, match="temporary failure"):
            await tracker.flush(db_session)
    assert private_ip(1004) in tracker._dirty
    await tracker.flush(db_session)
    assert not tracker._dirty
    assert (await tracker.get_all(db_session))[0]["request_count"] == 2


async def test_cancelled_flush_preserves_pending_records(db_session, monkeypatch):
    tracker = IPTrackerService()
    tracker.record("10.0.0.1")
    with monkeypatch.context() as patch:
        patch.setattr(db_session, "execute", AsyncMock(side_effect=asyncio.CancelledError()))
        with pytest.raises(asyncio.CancelledError):
            await tracker.flush(db_session)
    assert "10.0.0.1" in tracker._dirty
    await tracker.flush(db_session)
    assert len(await tracker.get_all(db_session)) == 1


async def test_evicted_ip_returning_before_flush_preserves_persisted_count(db_session):
    tracker = IPTrackerService()
    tracker.record(private_ip(0))
    tracker.record(private_ip(0))
    await tracker.flush(db_session)
    for i in range(1, 1001):
        tracker.record(private_ip(i))
    assert private_ip(0) not in tracker._ips
    tracker.record(private_ip(0))
    await tracker.flush(db_session)
    row = await db_session.scalar(select(ConnectedIP).where(ConnectedIP.ip_address == private_ip(0)))
    assert row.request_count == 3


async def test_tracker_flushes_add_counts_instead_of_overwriting(db_session):
    first, second = IPTrackerService(), IPTrackerService()
    first.record("10.0.0.1")
    first.record("10.0.0.1")
    second.record("10.0.0.1")
    latest = second._ips["10.0.0.1"]["last_seen"]
    await second.flush(db_session)
    await first.flush(db_session)
    row = (await second.get_all(db_session))[0]
    assert row["request_count"] == 3
    assert row["last_seen"] == latest


async def test_geoip_work_is_bounded_and_stopped(monkeypatch):
    tracker = IPTrackerService()
    event = asyncio.Event()

    async def lookup(ip):
        await event.wait()

    monkeypatch.setattr(tracker, "_lookup_country", lookup)
    for i in range(200):
        tracker.record(str(ip_address("8.8.0.1") + i))
    assert len(tracker._lookups) == tracker.MAX_GEOIP_LOOKUPS
    await tracker.close()
    assert not tracker._lookups


async def test_ordinary_404s_do_not_open_database_and_ban_at_threshold(db_session, monkeypatch):
    service = IPBanService()
    monkeypatch.setattr(settings, "IP_BAN_HIT_THRESHOLD", 5)
    session_factory = AsyncMock(side_effect=AssertionError("unnecessary DB session"))
    monkeypatch.setattr(ban_module, "async_session", session_factory)
    for _ in range(4):
        await service.record_404("8.8.8.8", path="/missing")
        assert not service.is_banned("8.8.8.8")
    session_factory.assert_not_called()
    await service.record_404("8.8.8.8", db_session, path="/missing")
    assert service.is_banned("8.8.8.8")
    row = await db_session.scalar(select(BannedIP))
    assert row.hit_count == 5


@pytest.mark.parametrize("path", ["/.env", "/.env.production", "/.git/config", "/WP-login.php", "/%2eenv"])
async def test_scanner_probes_ban_immediately_and_persist(db_session, path):
    service = IPBanService()
    await service.record_404("8.8.8.8", db_session, path=path)
    assert service.is_banned("8.8.8.8")
    assert (await db_session.scalar(select(BannedIP))).hit_count == 1
    restored = IPBanService()
    await restored.load_from_db(db_session)
    assert restored.is_banned("8.8.8.8")


async def test_expired_hits_do_not_count_and_hit_cache_is_bounded(monkeypatch):
    service = IPBanService()
    monkeypatch.setattr(settings, "IP_BAN_HIT_THRESHOLD", 5)
    current = datetime(2026, 1, 1, tzinfo=timezone.utc)

    class Clock:
        @staticmethod
        def now(tz):
            return current

    monkeypatch.setattr(ban_module, "datetime", Clock)
    for _ in range(4):
        await service.record_404("8.8.8.8", path="/missing")
    current += timedelta(seconds=settings.IP_BAN_WINDOW_SECONDS + 1)
    await service.record_404("8.8.8.8", path="/missing")
    assert len(service._404_hits["8.8.8.8"]) == 1
    for i in range(1200):
        await service.record_404(private_ip(i), path="/missing")
    assert len(service._404_hits) == 1000


async def test_benign_paths_and_loopback_do_not_ban(db_session):
    service = IPBanService()
    for _ in range(20):
        await service.record_404("8.8.8.8", path="/favicon.ico")
        await service.record_404("::ffff:127.0.0.1", path="/.env")
    assert not service._banned and not service._404_hits


async def test_concurrent_hits_persist_one_ban(monkeypatch):
    service = IPBanService()
    persist = AsyncMock()
    monkeypatch.setattr(service, "_persist_ban", persist)
    await asyncio.gather(*(service.record_404("8.8.8.8", object(), path="/.env") for _ in range(10)))
    assert persist.await_count == 1


async def test_known_resource_404_is_exempt_but_unknown_route_counts(monkeypatch):
    import main
    record = AsyncMock()
    monkeypatch.setattr(main.ip_ban_service, "record_404", record)
    app = FastAPI()
    app.middleware("http")(main.ip_ban_middleware)

    @app.get("/resources/{resource_id}")
    async def missing_resource(resource_id: str):
        return JSONResponse({"detail": "not found"}, status_code=404)

    async with AsyncClient(transport=ASGITransport(app=app, client=("10.0.0.5", 1234)), base_url="http://test") as client:
        assert (await client.get("/resources/missing")).status_code == 404
        record.assert_not_called()
        assert (await client.get("/unknown-route")).status_code == 404
    record.assert_awaited_once_with("10.0.0.5", path="/unknown-route")
