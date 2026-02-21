"""
Extracts Twitch activation codes from subprocess miner log files.
"""

import re
from pathlib import Path


def extract_twitch_activation(log_path: Path, lines: int = 10) -> dict[str, str | None]:
    """
    Extract Twitch activation URL and code from the last lines of a log file.
    Returns a dict with 'activation_url' and 'activation_code' (both None if not found).
    """
    result: dict[str, str | None] = {"activation_url": None, "activation_code": None}
    if not log_path or not log_path.exists():
        return result

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        log_lines = f.readlines()[-lines:]

    found_url = False
    found_code = None

    for line in log_lines:
        if re.search(r"https://www\.twitch\.tv/activate", line):
            found_url = True
        code_match = re.search(r"enter this code: ([A-Z0-9]{6,})", line)
        if code_match:
            found_code = code_match.group(1)

    if found_code:
        result["activation_code"] = found_code
        result["activation_url"] = "https://www.twitch.tv/activate" if found_url else None

    return result
