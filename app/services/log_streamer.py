"""
Tails miner log files and streams them to WebSocket clients.
"""

import asyncio
import json
from collections import deque
from pathlib import Path

from app.core.config import settings


def get_instance_log_file(instance_id: str) -> Path | None:
    """Return the miner log file for an instance: <twitch_username>.log only."""
    instance_dir = settings.INSTANCES_DIR / instance_id
    logs_dir = instance_dir / "logs"
    if not logs_dir.exists():
        return None

    config_file = instance_dir / "config.json"
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text(encoding="utf-8"))
            twitch_username = config.get("twitch_username")
            if twitch_username:
                expected = logs_dir / f"{twitch_username}.log"
                if expected.exists():
                    return expected
        except Exception:
            pass

    return None


async def tail_log(
    instance_id: str,
    history_lines: int | None = None,
):
    """
    Async generator that yields log lines from a miner instance.
    First yields the last `history_lines` lines, then follows new output.
    
    Args:
        instance_id: The instance ID
        history_lines: Number of historical lines to send (default: complete file)
    """
    # Wait for log file to exist (might take a moment after starting)
    sent_waiting = False
    for _ in range(30):  # Wait up to 30 seconds
        log_file = get_instance_log_file(instance_id)
        if log_file is not None and log_file.exists():
            break
        if not sent_waiting:
            yield "[system] Waiting for miner to produce output...\n"
            sent_waiting = True
        await asyncio.sleep(1)
    else:
        yield "[system] Log file not found. Is the miner running?\n"
        return

    # Send historical lines without loading full file into memory
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

        # Now follow new lines (like tail -f)
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                await asyncio.sleep(0.3)
