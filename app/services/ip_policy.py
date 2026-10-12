"""Shared bounds and address normalization for IP tracking and bans."""

from ipaddress import ip_address

IP_TRACKING_LIMIT = 1000


def normalize_ip(value: str) -> str | None:
    try:
        address = ip_address(value)
    except ValueError:
        return None
    address = getattr(address, "ipv4_mapped", None) or address
    if address.is_loopback or address.is_unspecified:
        return None
    return str(address)
