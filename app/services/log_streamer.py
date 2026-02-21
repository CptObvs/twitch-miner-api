"""
Tails subprocess miner log files and streams them as SSE lines.
"""

import asyncio
from collections import deque
from pathlib import Path

from app.core.config import settings


def get_log_file_path(instance_id: str, twitch_username: str) -> Path:
    """Return the expected log file path for a subprocess miner instance."""
    return settings.INSTANCES_DIR / instance_id / "logs" / f"{twitch_username}.log"


async def tail_log(
    instance_id: str,
    twitch_username: str,
    history_lines: int | None = None,
):
    """
    Async generator yielding log lines from a subprocess miner instance.
    First yields the last `history_lines` lines, then follows new output.
    """
    log_file = get_log_file_path(instance_id, twitch_username)

    sent_waiting = False
    for _ in range(30):
        if log_file.exists():
            break
        if not sent_waiting:
            yield "[system] Waiting for miner to produce output...\n"
            sent_waiting = True
        await asyncio.sleep(1)
    else:
        yield "[system] Log file not found. Is the miner running?\n"
        return

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        if history_lines is None:
            for line in f:
                yield line
        else:
            history = deque(maxlen=history_lines)
            for line in f:
                history.append(line)
            for line in history:
                yield line

        while True:
            line = f.readline()
            if line:
                yield line
            else:
                await asyncio.sleep(0.3)
