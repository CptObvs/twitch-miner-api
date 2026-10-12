"""
IP Ban Service
==============
Tracks unknown paths using a bounded sliding window. Scanner probes ban
immediately; other unknown paths ban after 5 hits in 60 seconds by default.
"""

import logging
from collections import OrderedDict, deque
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote
import uuid

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.sqlite import insert

from app.core.config import settings
from app.models.database import BannedIP, async_session
from app.services.ip_policy import IP_TRACKING_LIMIT, normalize_ip

logger = logging.getLogger("uvicorn.error")

_BENIGN_PATHS = {"/favicon.ico", "/robots.txt", "/sitemap.xml"}
_PROBE_ROOTS = ("/.env", "/.git", "/wp-login.php", "/wp-admin", "/xmlrpc.php", "/phpmyadmin")


def _is_probe(path: str) -> bool:
    path = unquote(path).lower().rstrip("/")
    return any(path == root or path.startswith(root + "/")
               or (root == "/.env" and path.startswith("/.env.")) for root in _PROBE_ROOTS)


class IPBanService:
    def __init__(self) -> None:
        self._banned: dict[str, datetime] = {}
        self._404_hits: OrderedDict[str, deque[datetime]] = OrderedDict()

    # ------------------------------------------------------------------
    # Startup

    async def load_from_db(self, db: AsyncSession) -> None:
        """Load active bans from DB into memory on startup."""
        now = datetime.now(timezone.utc)
        await db.execute(delete(BannedIP).where(BannedIP.banned_until <= now))
        await db.commit()
        self._banned.clear()
        self._404_hits.clear()
        result = await db.execute(select(BannedIP).where(BannedIP.banned_until > now))
        for row in result.scalars().all():
            banned_until = row.banned_until
            if banned_until.tzinfo is None:
                banned_until = banned_until.replace(tzinfo=timezone.utc)
            if banned_until > now:
                self._banned[row.ip_address] = banned_until
        logger.info("IP ban service: loaded %d active ban(s) from DB", len(self._banned))

    # ------------------------------------------------------------------
    # Check

    def is_banned(self, ip: str) -> bool:
        """Sync check — returns True if the IP is currently banned."""
        if (ip := normalize_ip(ip)) is None:
            return False
        banned_until = self._banned.get(ip)
        if banned_until is None:
            return False
        now = datetime.now(timezone.utc)
        if banned_until.tzinfo is None:
            banned_until = banned_until.replace(tzinfo=timezone.utc)
        if now >= banned_until:
            del self._banned[ip]
            return False
        return True

    # ------------------------------------------------------------------
    # Record

    async def record_404(self, ip: str, db: AsyncSession | None = None, *, path: str = "") -> None:
        """Record a 404 hit for an IP. Bans the IP if the threshold is reached."""
        if (ip := normalize_ip(ip)) is None or path in _BENIGN_PATHS or self.is_banned(ip):
            return

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=settings.IP_BAN_WINDOW_SECONDS)
        # Ordered by the most recent hit, so old entries can be pruned cheaply.
        while self._404_hits:
            oldest = next(iter(self._404_hits))
            if self._404_hits[oldest][-1] >= window_start:
                break
            self._404_hits.popitem(last=False)
        if ip not in self._404_hits:
            if len(self._404_hits) >= IP_TRACKING_LIMIT:
                self._404_hits.popitem(last=False)
            self._404_hits[ip] = deque()
        hits = self._404_hits[ip]
        self._404_hits.move_to_end(ip)

        # Prune hits outside the window
        while hits and hits[0] < window_start:
            hits.popleft()

        hits.append(now)

        if _is_probe(path) or len(hits) >= settings.IP_BAN_HIT_THRESHOLD:
            banned_until = now + timedelta(hours=settings.IP_BAN_DURATION_HOURS)
            hit_count = len(hits)
            self._banned[ip] = banned_until
            self._404_hits.pop(ip, None)
            # Ordinary 404s require no database session or write.
            if db is not None:
                await self._persist_ban(ip, now, banned_until, hit_count, db)
            else:
                async with async_session() as ban_db:
                    await self._persist_ban(ip, now, banned_until, hit_count, ban_db)
            logger.warning(
                "IP ban service: banned %s for %dh after %d 404 hits in %ds",
                ip, settings.IP_BAN_DURATION_HOURS, hit_count, settings.IP_BAN_WINDOW_SECONDS,
            )

    async def _persist_ban(
        self,
        ip: str,
        banned_at: datetime,
        banned_until: datetime,
        hit_count: int,
        db: AsyncSession,
    ) -> None:
        stmt = insert(BannedIP).values(
            id=str(uuid.uuid4()), ip_address=ip, banned_at=banned_at,
            banned_until=banned_until, hit_count=hit_count,
        )
        await db.execute(stmt.on_conflict_do_update(
            index_elements=[BannedIP.ip_address],
            set_={"banned_at": banned_at, "banned_until": banned_until, "hit_count": hit_count},
        ))
        await db.commit()

    # ------------------------------------------------------------------
    # Admin helpers

    async def get_active_bans(self, db: AsyncSession) -> list[BannedIP]:
        """Return all bans whose banned_until is in the future."""
        now = datetime.now(timezone.utc)
        result = await db.execute(select(BannedIP).where(BannedIP.banned_until > now))
        return list(result.scalars().all())

    async def ban(self, ip: str, duration_hours: int, db: AsyncSession) -> None:
        """Manually ban an IP for the given duration."""
        if (ip := normalize_ip(ip)) is None:
            return
        now = datetime.now(timezone.utc)
        banned_until = now + timedelta(hours=duration_hours)
        self._banned[ip] = banned_until
        self._404_hits.pop(ip, None)
        await self._persist_ban(ip, now, banned_until, 0, db)
        logger.info("IP ban service: manually banned %s for %dh", ip, duration_hours)

    async def unban(self, ip: str, db: AsyncSession) -> bool:
        """Remove a ban for the given IP. Returns True if a ban was found."""
        if (ip := normalize_ip(ip)) is None:
            return False
        result = await db.execute(select(BannedIP).where(BannedIP.ip_address == ip))
        row = result.scalar_one_or_none()
        if row is None:
            return False
        await db.delete(row)
        await db.commit()
        self._banned.pop(ip, None)
        self._404_hits.pop(ip, None)
        return True


ip_ban_service = IPBanService()
