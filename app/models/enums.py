"""
Enum definitions for the application.
"""

from enum import Enum


class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"


class InstanceState(str, Enum):
    """Miner instance lifecycle states."""
    STOPPED = "stopped"
    RUNNING = "running"
    STOPPING = "stopping"
