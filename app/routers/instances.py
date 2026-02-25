import asyncio
import json
import logging
import re
import shutil
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.database import User, MinerInstance, get_db, async_session
from app.models.enums import InstanceState, MinerType
from app.models.schemas import (
    InstanceCreate,
    InstanceResponse,
    InstanceStatus,
    StreamerPointsSnapshot,
    StreamersUpdate,
)
from app.services.auth import get_current_user, verify_token, oauth2_scheme
from app.services.miner_manager import miner_manager
from app.services.activation_log_parser import extract_twitch_activation_from_lines
from app.services.points import extract_points_from_lines
from app.services.socket_manager import push_instance_update

router = APIRouter(prefix="/instances", tags=["instances"])
logger = logging.getLogger("uvicorn.error")

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mGKH]")

# ------------------------------------------------------------------
# In-memory caches (avoid repeated log reads)
# ------------------------------------------------------------------

_POINTS_CACHE_TTL = 30.0   # seconds
_ACTIVATION_CACHE_TTL = 60.0  # seconds

# instance_id -> (timestamp, data)
_points_cache: dict[str, tuple[float, list]] = {}
_activation_cache: dict[str, tuple[float, dict]] = {}


def _invalidate_instance_caches(instance_id: str) -> None:
    _points_cache.pop(instance_id, None)
    _activation_cache.pop(instance_id, None)


# ------------------------------------------------------------------
# Config helpers (V2/subprocess only)
# ------------------------------------------------------------------

def _read_config(instance_id: str) -> dict:
    config_file = settings.INSTANCES_DIR / instance_id / "config.json"
    if not config_file.exists():
        raise HTTPException(400, "Instance config not found. Recreate the instance.")
    return json.loads(config_file.read_text(encoding="utf-8"))


def _write_config(instance_id: str, data: dict) -> None:
    config_dir = settings.INSTANCES_DIR / instance_id
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps(data), encoding="utf-8")


# ------------------------------------------------------------------
# Response builder
# ------------------------------------------------------------------

def _instance_to_response(
    instance: MinerInstance,
    activation: dict[str, str | None] | None = None,
) -> InstanceResponse:
    ui_url = None
    streamers: list[str] = []

    if instance.miner_type == MinerType.TwitchDropsMiner:
        if instance.status == InstanceState.RUNNING:
            ui_url = f"/instances/{instance.id}/ui"
    elif instance.miner_type == MinerType.TwitchPointsMinerV2:
        try:
            config = _read_config(instance.id)
            streamers = config.get("streamers", [])
        except Exception:
            pass

    act = activation or {"activation_code": None, "activation_url": None}

    return InstanceResponse(
        id=instance.id,
        user_id=instance.user_id,
        miner_type=instance.miner_type,
        status=instance.status,
        container_id=instance.container_id,
        port=instance.port,
        ui_url=ui_url,
        twitch_username=instance.twitch_username,
        activation_code=act["activation_code"],
        activation_url=act["activation_url"],
        streamers=streamers,
        created_at=instance.created_at,
        last_started_at=instance.last_started_at,
        last_stopped_at=instance.last_stopped_at,
    )


async def _fetch_activation(instance: MinerInstance) -> dict[str, str | None]:
    """Fetch activation code from recent logs for a V2 instance (works even when stopped)."""
    result: dict[str, str | None] = {"activation_url": None, "activation_code": None}
    if instance.miner_type != MinerType.TwitchPointsMinerV2:
        return result

    cached = _activation_cache.get(instance.id)
    if cached and time.monotonic() - cached[0] < _ACTIVATION_CACHE_TTL:
        return cached[1]

    # For V2 instances, always try to read from output.log file regardless of container status
    lines = await miner_manager.get_recent_logs(
        instance.container_id or "",
        tail=20,
        instance_id=instance.id,
        instance_type=instance.miner_type
    )
    result = extract_twitch_activation_from_lines(lines)
    _activation_cache[instance.id] = (time.monotonic(), result)
    return result


# ------------------------------------------------------------------
# Auth helper
# ------------------------------------------------------------------

async def _get_user_instance(
    instance_id: str,
    current_user: User,
    db: AsyncSession,
) -> MinerInstance:
    if current_user.is_admin():
        result = await db.execute(
            select(MinerInstance).where(MinerInstance.id == instance_id)
        )
    else:
        result = await db.execute(
            select(MinerInstance).where(
                MinerInstance.id == instance_id,
                MinerInstance.user_id == current_user.id,
            )
        )
    instance = result.scalar_one_or_none()
    if not instance:
        raise HTTPException(404, "Instance not found")
    return instance


# ------------------------------------------------------------------
# CRUD
# ------------------------------------------------------------------

@router.post("/", response_model=InstanceResponse, status_code=201)
async def create_instance(
    data: InstanceCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Create a new instance (DB record only, no container started)."""
    if not current_user.is_admin():
        result = await db.execute(
            select(MinerInstance).where(MinerInstance.user_id == current_user.id)
        )
        if len(result.scalars().all()) >= settings.MAX_INSTANCES_PER_USER:
            raise HTTPException(
                400, f"Maximum {settings.MAX_INSTANCES_PER_USER} instances per user"
            )

    instance = MinerInstance(
        user_id=current_user.id,
        miner_type=data.miner_type,
        twitch_username=data.twitch_username,
    )
    db.add(instance)
    await db.commit()
    await db.refresh(instance)

    if data.miner_type == MinerType.TwitchPointsMinerV2:
        _write_config(instance.id, {"streamers": data.streamers})

    logger.info(
        "Instance created: id=%s user_id=%s type=%s",
        instance.id, current_user.id, data.miner_type,
    )
    return _instance_to_response(instance)


@router.get("/", response_model=list[InstanceResponse])
async def list_instances(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """List instances — admin sees all, users see their own."""
    if current_user.is_admin():
        result = await db.execute(select(MinerInstance))
    else:
        result = await db.execute(
            select(MinerInstance).where(MinerInstance.user_id == current_user.id)
        )
    return [_instance_to_response(inst) for inst in result.scalars().all()]


@router.get("/{instance_id}", response_model=InstanceResponse)
async def get_instance(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    instance = await _get_user_instance(instance_id, current_user, db)
    activation = await _fetch_activation(instance)
    return _instance_to_response(instance, activation)


@router.delete("/{instance_id}", status_code=204)
async def delete_instance(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Stop container (if running), delete DB record and data directory."""
    instance = await _get_user_instance(instance_id, current_user, db)

    if instance.status != InstanceState.STOPPED:
        await miner_manager.stop(instance_id, db_session=db)

    await db.delete(instance)
    await db.commit()

    shutil.rmtree(settings.INSTANCES_DIR / instance_id, ignore_errors=True)
    logger.info("Instance deleted: id=%s user_id=%s", instance_id, current_user.id)


# ------------------------------------------------------------------
# Start / Stop
# ------------------------------------------------------------------

@router.post("/{instance_id}/start", response_model=InstanceStatus)
async def start_instance(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Start the Docker container for an instance."""
    instance = await _get_user_instance(instance_id, current_user, db)

    if instance.status != InstanceState.STOPPED:
        raise HTTPException(409, f"Instance is already {instance.status.value}")

    if not current_user.is_admin():
        result = await db.execute(
            select(MinerInstance).where(
                MinerInstance.user_id == current_user.id,
                MinerInstance.status == InstanceState.RUNNING,
            )
        )
        if len(result.scalars().all()) >= settings.MAX_INSTANCES_PER_USER:
            raise HTTPException(
                400, f"Maximum {settings.MAX_INSTANCES_PER_USER} running instances per user"
            )

    streamers: list[str] = []
    if instance.miner_type == MinerType.TwitchPointsMinerV2:
        config = _read_config(instance_id)
        streamers = config.get("streamers", [])

    try:
        container_id = await miner_manager.start(
            instance_id,
            miner_type=instance.miner_type,
            twitch_username=instance.twitch_username,
            streamers=streamers,
            db_session=db,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    await db.refresh(instance)

    # Invalidate caches so next read fetches fresh data after container start
    _invalidate_instance_caches(instance_id)

    activation: dict[str, str | None] = {"activation_url": None, "activation_code": None}
    if instance.miner_type == MinerType.TwitchPointsMinerV2:
        await asyncio.sleep(1)
        lines = await miner_manager.get_recent_logs(
            container_id, tail=20, instance_id=instance_id, instance_type=instance.miner_type
        )
        activation = extract_twitch_activation_from_lines(lines)

    logger.info(
        "Instance started: id=%s type=%s container=%s",
        instance_id, instance.miner_type, container_id[:12],
    )
    status_response = InstanceStatus(
        id=instance_id,
        status=InstanceState.RUNNING,
        container_id=container_id,
        port=instance.port,
        activation_url=activation["activation_url"],
        activation_code=activation["activation_code"],
    )
    asyncio.create_task(push_instance_update(
        current_user.id,
        {**status_response.model_dump(), "user_id": current_user.id},
    ))
    return status_response


@router.post("/{instance_id}/stop", response_model=InstanceStatus)
async def stop_instance(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Stop the container. Blocks until fully stopped."""
    instance = await _get_user_instance(instance_id, current_user, db)
    logger.info("Stop requested: id=%s user_id=%s", instance_id, current_user.id)

    await miner_manager.stop(instance_id, db_session=db)
    _invalidate_instance_caches(instance_id)
    status_response = InstanceStatus(id=instance_id, status=InstanceState.STOPPED)
    asyncio.create_task(push_instance_update(
        current_user.id,
        {**status_response.model_dump(), "user_id": current_user.id},
    ))
    return status_response


# ------------------------------------------------------------------
# Status
# ------------------------------------------------------------------

@router.get("/{instance_id}/status", response_model=InstanceStatus)
async def instance_status(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Live-check container status against Docker runtime, then return DB state."""
    instance = await _get_user_instance(instance_id, current_user, db)

    await miner_manager.reconcile_instance_status(instance_id, db_session=db)
    await db.refresh(instance)

    activation = await _fetch_activation(instance)
    return InstanceStatus(
        id=instance_id,
        status=instance.status,
        container_id=instance.container_id,
        port=instance.port,
        activation_url=activation["activation_url"],
        activation_code=activation["activation_code"],
    )


# ------------------------------------------------------------------
# Logs (SSE)
# ------------------------------------------------------------------

@router.get("/{instance_id}/logs")
async def stream_logs(
    instance_id: str,
    token: str = Depends(oauth2_scheme),
    tail: int = Query(settings.LOG_HISTORY_LINES, ge=1),
    full: bool = Query(False),
):
    """
    SSE endpoint — streams instance logs in real-time via docker logs.
    Pass full=true to load the entire log history instead of the last `tail` lines.

    The DB session is intentionally closed before streaming starts so that no
    connection-pool slot is held for the (potentially unbounded) stream duration.

    curl:  curl -N -H "Authorization: Bearer <token>" http://localhost:8000/api/instances/{id}/logs
    """
    async with async_session() as db:
        current_user = await verify_token(token, db)
        instance = await _get_user_instance(instance_id, current_user, db)
        container_id = instance.container_id
        user_id = current_user.id
    # DB session is now fully closed — no pool slot held during streaming

    # For V2 instances, allow log streaming even when container is stopped
    if instance.miner_type == MinerType.TwitchPointsMinerV2:
        # Always allow access to V2 log files regardless of container status
        pass
    elif not container_id:
        raise HTTPException(409, "Instance has no running container")

    logger.info("Log stream opened: id=%s user_id=%s full=%s", instance_id, user_id, full)

    tail_arg = "all" if full else str(tail)

    async def event_generator():
        if instance.miner_type == MinerType.TwitchPointsMinerV2:
            # For V2 instances, read from output.log file
            log_file = settings.INSTANCES_DIR / instance_id / "output.log"
            
            # First, send the existing log content (tail or full)
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Send existing lines
                    start_idx = 0 if full else max(0, len(lines) - tail)
                    for line in lines[start_idx:]:
                        line = line.rstrip('\n')
                        if line:
                            yield f"data: {line}\n\n"
                except Exception:
                    pass
            
            # Then monitor the file for new content
            try:
                # Wait up to 10 s for the file to appear (race: container not yet started)
                for _ in range(100):
                    if log_file.exists():
                        break
                    await asyncio.sleep(0.1)

                if not log_file.exists():
                    return

                with open(log_file, 'r', encoding='utf-8') as f:
                    # Move to end of file for real-time monitoring
                    f.seek(0, 2)  # Seek to end

                    while True:
                        line = f.readline()
                        if line:
                            line = line.rstrip('\n')
                            if line:
                                yield f"data: {line}\n\n"
                        else:
                            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except asyncio.CancelledError:
                pass
        else:
            # For non-V2 instances, use Docker logs as before
            proc = await asyncio.create_subprocess_exec(
                "docker", "logs", "-f",
                "--tail", tail_arg,
                container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                async for raw in proc.stdout:
                    line = ANSI_ESCAPE.sub("", raw.decode(errors="replace")).strip()
                    if line:
                        yield f"data: {line}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                proc.kill()
                await proc.wait()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ------------------------------------------------------------------
# V2-only: Streamers
# ------------------------------------------------------------------

@router.get("/{instance_id}/streamers")
async def get_streamers(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get the streamer list for a V2 instance."""
    instance = await _get_user_instance(instance_id, current_user, db)
    if instance.miner_type != MinerType.TwitchPointsMinerV2:
        raise HTTPException(400, "Streamers are only available for V2 instances")
    config = _read_config(instance_id)
    return {"streamers": config.get("streamers", [])}


@router.put("/{instance_id}/streamers")
async def update_streamers(
    instance_id: str,
    data: StreamersUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Update the streamer list. Restart the instance for changes to take effect."""
    instance = await _get_user_instance(instance_id, current_user, db)
    if instance.miner_type != MinerType.TwitchPointsMinerV2:
        raise HTTPException(400, "Streamers are only available for V2 instances")
    config = _read_config(instance_id)
    config["streamers"] = data.streamers
    _write_config(instance_id, config)
    logger.info("Streamers updated: id=%s count=%d", instance_id, len(data.streamers))
    return {"streamers": data.streamers}


# ------------------------------------------------------------------
# V2-only: Points
# ------------------------------------------------------------------

@router.get("/{instance_id}/points", response_model=list[StreamerPointsSnapshot])
async def get_instance_points_route(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Latest channel points per streamer for a V2 instance."""
    instance = await _get_user_instance(instance_id, current_user, db)
    if instance.miner_type != MinerType.TwitchPointsMinerV2:
        raise HTTPException(400, "Points tracking is only available for V2 instances")

    configured: list[str] = []
    try:
        config = _read_config(instance_id)
        configured = [str(s).strip() for s in config.get("streamers", []) if str(s).strip()]
    except Exception:
        pass

    cached = _points_cache.get(instance_id)
    if cached and time.monotonic() - cached[0] < _POINTS_CACHE_TTL:
        snapshots = cached[1]
    else:
        lines: list[str] = []
        # For V2 instances, always try to read from output.log file regardless of container status
        if instance.miner_type == MinerType.TwitchPointsMinerV2:
            lines = await miner_manager.get_recent_logs(
                instance.container_id or "",
                tail=2000,
                instance_id=instance.id,
                instance_type=instance.miner_type
            )

        points = extract_points_from_lines(
            lines,
            expected_streamers=set(configured) if configured else None,
        )
        snapshots = [
            StreamerPointsSnapshot(streamer=name, channel_points=points.get(name.lower()))
            for name in configured
        ]
        _points_cache[instance_id] = (time.monotonic(), snapshots)

    return snapshots
