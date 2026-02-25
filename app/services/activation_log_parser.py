"""
Extracts Twitch activation codes from miner log lines.
"""

import re

_RE_ACTIVATION_URL = re.compile(r"https://www\.twitch\.tv/activate")
_RE_ACTIVATION_CODE = re.compile(r"enter this code: ([A-Z0-9]{6,})")


def extract_twitch_activation_from_lines(lines: list[str]) -> dict[str, str | None]:
    """
    Extract Twitch activation URL and code from a list of log lines.
    Returns a dict with 'activation_url' and 'activation_code' (both None if not found).
    """
    result: dict[str, str | None] = {"activation_url": None, "activation_code": None}

    found_url = False
    found_code = None

    for line in lines:
        if _RE_ACTIVATION_URL.search(line):
            found_url = True
        code_match = _RE_ACTIVATION_CODE.search(line)
        if code_match:
            found_code = code_match.group(1)

    if found_code:
        result["activation_code"] = found_code
        result["activation_url"] = "https://www.twitch.tv/activate" if found_url else None

    return result
