"""Keep the 1,000 most recently active IPs; batch persistence and bound GeoIP work."""

import asyncio
import json
import logging
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from ipaddress import ip_address

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ip_policy import IP_TRACKING_LIMIT, normalize_ip

logger = logging.getLogger("uvicorn.error")

_PRUNE_SQL = text("""
    DELETE FROM connected_ips WHERE ip_address NOT IN (
        SELECT ip_address FROM connected_ips
        ORDER BY last_seen DESC, ip_address DESC LIMIT :limit
    )
""")
_SELECT_SQL = text("""
    SELECT ip_address, country, country_code, first_seen, last_seen, request_count
    FROM connected_ips ORDER BY last_seen DESC, ip_address DESC LIMIT :limit
""")
_UPSERT_SQL = text("""
    INSERT INTO connected_ips
        (ip_address, country, country_code, first_seen, last_seen, request_count)
    VALUES (:ip_address, :country, :country_code, :first_seen, :last_seen, :request_count)
    ON CONFLICT(ip_address) DO UPDATE SET
        last_seen = MAX(connected_ips.last_seen, excluded.last_seen),
        request_count = connected_ips.request_count + excluded.request_count,
        country = COALESCE(excluded.country, connected_ips.country),
        country_code = COALESCE(excluded.country_code, connected_ips.country_code)
""")


class IPTrackerService:
    MAX_GEOIP_LOOKUPS = 8

    def __init__(self) -> None:
        self._ips: OrderedDict[str, dict] = OrderedDict()
        self._dirty: set[str] = set()
        self._pending_counts: dict[str, int] = {}
        self._lookups: dict[str, asyncio.Task] = {}
        self._flush_lock = asyncio.Lock()

    async def load_from_db(self, db: AsyncSession) -> None:
        await db.execute(_PRUNE_SQL, {"limit": IP_TRACKING_LIMIT})
        await db.commit()
        rows = (await db.execute(_SELECT_SQL, {"limit": IP_TRACKING_LIMIT})).mappings().all()
        self._ips = OrderedDict((row["ip_address"], dict(row)) for row in reversed(rows))
        self._dirty.clear()
        self._pending_counts.clear()
        logger.debug("IPTracker: loaded %d IPs from DB", len(self._ips))

    def record(self, ip: str) -> None:
        if (ip := normalize_ip(ip)) is None:
            return
        now = datetime.now(timezone.utc).isoformat(sep=" ")
        if ip not in self._ips:
            if len(self._ips) >= IP_TRACKING_LIMIT:
                evicted, _ = self._ips.popitem(last=False)
                self._dirty.discard(evicted)
                self._pending_counts.pop(evicted, None)
            self._ips[ip] = {
                "ip_address": ip, "country": None, "country_code": None,
                "first_seen": now, "last_seen": now, "request_count": 1,
            }
        else:
            self._ips[ip]["last_seen"] = now
            self._ips[ip]["request_count"] += 1
            self._ips.move_to_end(ip)
        self._dirty.add(ip)
        self._pending_counts[ip] = self._pending_counts.get(ip, 0) + 1
        if (self._ips[ip]["country"] is None and ip_address(ip).is_global
                and ip not in self._lookups and len(self._lookups) < self.MAX_GEOIP_LOOKUPS):
            self._lookups[ip] = asyncio.create_task(self._lookup_country(ip))

    async def flush(self, db: AsyncSession) -> None:
        async with self._flush_lock:
            if not self._dirty:
                return
            dirty = self._dirty.copy()
            self._dirty.difference_update(dirty)
            # Snapshot values before awaiting: new requests remain dirty for the next flush.
            counts = {ip: self._pending_counts.pop(ip, 0) for ip in dirty}
            entries = [dict(self._ips[ip], request_count=counts[ip]) for ip in dirty if ip in self._ips]
            try:
                if entries:
                    await db.execute(_UPSERT_SQL, entries)
                await db.execute(_PRUNE_SQL, {"limit": IP_TRACKING_LIMIT})
                await db.commit()
            except BaseException:
                await db.rollback()
                self._dirty.update(ip for ip in dirty if ip in self._ips)
                for ip, count in counts.items():
                    if ip in self._ips:
                        self._pending_counts[ip] = self._pending_counts.get(ip, 0) + count
                raise

    async def get_all(self, db: AsyncSession) -> list[dict]:
        rows = (await db.execute(_SELECT_SQL, {"limit": IP_TRACKING_LIMIT})).mappings().all()
        return [dict(row) for row in rows]

    async def close(self) -> None:
        tasks = list(self._lookups.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._lookups.clear()

    async def _lookup_country(self, ip: str) -> None:
        entry = self._ips.get(ip)
        try:
            data = await asyncio.to_thread(self._fetch_geoip_sync, ip)
            # A delayed result must not update an evicted/recreated record.
            if data and entry is not None and self._ips.get(ip) is entry:
                entry["country"] = data.get("country")
                entry["country_code"] = data.get("country_code")
                self._dirty.add(ip)
        except Exception as exc:
            logger.debug("IPTracker: GeoIP lookup failed for %s: %s", ip, exc)
        finally:
            self._lookups.pop(ip, None)

    @staticmethod
    def _fetch_geoip_sync(ip: str) -> dict | None:
        req = urllib.request.Request(
            f"https://ipwho.is/{ip}", headers={"User-Agent": "TwitchMinerAPI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        if data.get("success"):
            return {"country": data.get("country"), "country_code": data.get("country_code")}
        return None


ip_tracker_service = IPTrackerService()
