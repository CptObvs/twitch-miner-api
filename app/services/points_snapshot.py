"""
Scans subprocess miner log files to extract channel points per streamer.
Uses a short-lived cache to avoid repeated file reads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from app.services.log_streamer import get_log_file_path

STREAMER_DETAILS_RE = re.compile(r"Streamer\(username=([^,]+),[^)]*channel_points=([^)]+)\)")
STREAMER_INLINE_RE = re.compile(r"\s•\s([^()]+?)\s\(([^)]+)\)$")


@dataclass
class _CacheEntry:
    timestamp: float
    history_lines: int
    points_by_streamer: dict[str, str]


_cache: dict[str, _CacheEntry] = {}


def _iter_lines_reverse(file_path: Path, chunk_size: int = 8192):
    with open(file_path, "rb") as f:
        f.seek(0, 2)
        position = f.tell()
        buffer = b""
        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            f.seek(position)
            chunk = f.read(read_size)
            buffer = chunk + buffer
            parts = buffer.split(b"\n")
            buffer = parts[0]
            for raw in reversed(parts[1:]):
                if raw:
                    yield raw.decode("utf-8", errors="replace")
        if buffer:
            yield buffer.decode("utf-8", errors="replace")


def _extract_snapshot(line: str) -> tuple[str, str] | None:
    m = STREAMER_DETAILS_RE.search(line)
    if m:
        username = m.group(1).strip().lower()
        points = m.group(2).strip()
        if points:
            return username, points

    m = STREAMER_INLINE_RE.search(line)
    if m:
        username = m.group(1).strip().lower()
        points = m.group(2).strip()
        if points:
            return username, points

    return None


def _collect(
    instance_id: str,
    twitch_username: str,
    history_lines: int,
    expected_streamers: set[str] | None,
) -> dict[str, str]:
    log_file = get_log_file_path(instance_id, twitch_username)
    if not log_file.exists():
        return {}

    points: dict[str, str] = {}
    expected = {n.strip().lower() for n in (expected_streamers or set()) if n.strip()}
    target = len(expected)
    scanned = 0

    for line in _iter_lines_reverse(log_file):
        scanned += 1
        if scanned > history_lines:
            break
        snap = _extract_snapshot(line)
        if snap is None:
            continue
        name, pts = snap
        if name in points:
            continue
        if expected and name not in expected:
            continue
        points[name] = pts
        if target and len(points) >= target:
            break

    return points


def get_instance_points_snapshot(
    instance_id: str,
    twitch_username: str,
    *,
    history_lines: int = 2000,
    refresh: bool = False,
    max_age_seconds: int = 30,
    expected_streamers: set[str] | None = None,
) -> dict[str, str]:
    now = monotonic()
    cached = _cache.get(instance_id)

    if (
        not refresh
        and cached is not None
        and cached.history_lines == history_lines
        and now - cached.timestamp <= max_age_seconds
    ):
        data = cached.points_by_streamer
        if expected_streamers:
            exp = {n.strip().lower() for n in expected_streamers if n.strip()}
            return {k: v for k, v in data.items() if k in exp}
        return dict(data)

    fresh = _collect(instance_id, twitch_username, history_lines, expected_streamers)
    _cache[instance_id] = _CacheEntry(
        timestamp=now, history_lines=history_lines, points_by_streamer=dict(fresh)
    )
    return fresh
