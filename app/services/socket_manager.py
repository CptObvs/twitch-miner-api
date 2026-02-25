"""
Socket.IO server for real-time instance status push to connected clients.

Each authenticated user joins room 'user:{user_id}'.
Admins additionally join room 'admin' and receive all updates.
"""

import logging

import socketio

from app.core.config import settings
from app.models.database import async_session
from app.services.auth import verify_token

logger = logging.getLogger("uvicorn.error")

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=settings.CORS_ORIGINS,
    logger=False,
    engineio_logger=False,
)


@sio.event
async def connect(sid: str, environ: dict, auth: dict | None) -> None:
    token = (auth or {}).get("token")
    if not token:
        raise ConnectionRefusedError("authentication required")

    try:
        async with async_session() as db:
            user = await verify_token(token, db)
    except Exception:
        raise ConnectionRefusedError("invalid or expired token")

    await sio.enter_room(sid, f"user:{user.id}")
    if user.is_admin():
        await sio.enter_room(sid, "admin")

    logger.info("Socket.IO connect: sid=%s user_id=%s admin=%s", sid, user.id, user.is_admin())


@sio.event
async def disconnect(sid: str) -> None:
    logger.info("Socket.IO disconnect: sid=%s", sid)


async def push_instance_update(user_id: str, instance_data: dict) -> None:
    """Emit an instance_update event to the user's room and the admin room."""
    await sio.emit("instance_update", instance_data, room=f"user:{user_id}")
    await sio.emit("instance_update", instance_data, room="admin")
