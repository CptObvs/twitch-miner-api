import asyncio
import json
import logging
import re
import shutil
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.database import User, MinerInstance, get_db
from app.models.enums import InstanceState
from app.models.schemas import (
    InstanceCreate,
    InstancePointsSnapshotResponse,
    InstanceResponse,
    InstanceStatus,
    StreamerPointsSnapshot,
    StreamersUpdate,
)
from app.services.auth import get_current_user
from app.services.miner_manager import miner_manager as docker_manager
from app.services.process_manager import process_manager
from app.services.log_streamer import tail_log, get_log_file_path
from app.services.activation_log_parser import extract_twitch_activation
from app.services.points_snapshot import get_instance_points_snapshot

router = APIRouter(prefix="/instances", tags=["instances"])
logger = logging.getLogger("uvicorn.error")

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mGKH]")


# ------------------------------------------------------------------
# Config helpers (subprocess only)
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

def _activation_for(instance: MinerInstance) -> dict[str, str | None]:
    if (
        instance.miner_type != "subprocess"
        or instance.status != InstanceState.RUNNING
        or not instance.twitch_username
    ):
        return {"activation_url": None, "activation_code": None}
    log_path = get_log_file_path(instance.id, instance.twitch_username)
    return extract_twitch_activation(log_path)


def _instance_to_response(instance: MinerInstance) -> InstanceResponse:
    ui_url = None
    streamers: list[str] = []
    activation: dict[str, str | None] = {"activation_url": None, "activation_code": None}

    if instance.miner_type == "docker":
        if instance.status == InstanceState.RUNNING:
            ui_url = f"/instances/{instance.id}/ui"
    else:
        try:
            config = _read_config(instance.id)
            streamers = config.get("streamers", [])
        except Exception:
            pass
        activation = _activation_for(instance)

    return InstanceResponse(
        id=instance.id,
        user_id=instance.user_id,
        miner_type=instance.miner_type,
        status=instance.status,
        container_id=instance.container_id,
        port=instance.port,
        ui_url=ui_url,
        twitch_username=instance.twitch_username,
        pid=instance.pid,
        enable_analytics=instance.enable_analytics,
        analytics_port=instance.analytics_port,
        activation_code=activation["activation_code"],
        activation_url=activation["activation_url"],
        streamers=streamers,
        created_at=instance.created_at,
        last_started_at=instance.last_started_at,
        last_stopped_at=instance.last_stopped_at,
    )


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
    """Create a new instance (DB record only, no container/process started)."""
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
        enable_analytics=data.enable_analytics,
    )
    db.add(instance)
    await db.commit()
    await db.refresh(instance)

    if data.miner_type == "subprocess":
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
    return _instance_to_response(instance)


@router.delete("/{instance_id}", status_code=204)
async def delete_instance(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Stop container/process (if running), delete DB record and data directory."""
    instance = await _get_user_instance(instance_id, current_user, db)

    if instance.status != InstanceState.STOPPED:
        if instance.miner_type == "docker":
            await docker_manager.stop(instance_id, db_session=db)
        else:
            await process_manager.stop(instance_id, db_session=db)

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
    """Start the Docker container or Python subprocess for an instance."""
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

    if instance.miner_type == "docker":
        try:
            container_id = await docker_manager.start(instance_id, db_session=db)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        await db.refresh(instance)
        logger.info("Docker instance started: id=%s container=%s", instance_id, container_id[:12])
        return InstanceStatus(
            id=instance_id,
            status=InstanceState.RUNNING,
            container_id=container_id,
            port=instance.port,
        )
    else:
        config = _read_config(instance_id)
        try:
            pid = await process_manager.start(
                instance_id=instance_id,
                twitch_username=instance.twitch_username,
                streamers=config.get("streamers", []),
                enable_analytics=instance.enable_analytics,
                db_session=db,
            )
        except RuntimeError as e:
            raise HTTPException(409, str(e))

        await asyncio.sleep(1)
        log_path = get_log_file_path(instance_id, instance.twitch_username)
        activation = extract_twitch_activation(log_path)
        logger.info("Subprocess instance started: id=%s pid=%d", instance_id, pid)
        return InstanceStatus(
            id=instance_id,
            status=InstanceState.RUNNING,
            pid=pid,
            activation_url=activation["activation_url"],
            activation_code=activation["activation_code"],
        )


@router.post("/{instance_id}/stop", response_model=InstanceStatus)
async def stop_instance(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Stop the container/process. Blocks until fully stopped."""
    instance = await _get_user_instance(instance_id, current_user, db)
    logger.info("Stop requested: id=%s user_id=%s", instance_id, current_user.id)

    if instance.miner_type == "docker":
        await docker_manager.stop(instance_id, db_session=db)
    else:
        await process_manager.stop(instance_id, db_session=db)

    return InstanceStatus(id=instance_id, status=InstanceState.STOPPED)


# ------------------------------------------------------------------
# Status
# ------------------------------------------------------------------

@router.get("/{instance_id}/status", response_model=InstanceStatus)
async def instance_status(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Live-check container/process status against runtime, then return DB state."""
    instance = await _get_user_instance(instance_id, current_user, db)

    if instance.miner_type == "docker":
        await docker_manager.reconcile_instance_status(instance_id, db_session=db)
        await db.refresh(instance)
        return InstanceStatus(
            id=instance_id,
            status=instance.status,
            container_id=instance.container_id,
            port=instance.port,
        )
    else:
        await process_manager.reconcile_instance_status(instance_id, db_session=db)
        await db.refresh(instance)
        activation = _activation_for(instance)
        return InstanceStatus(
            id=instance_id,
            status=instance.status,
            pid=instance.pid,
            activation_url=activation["activation_url"],
            activation_code=activation["activation_code"],
        )


# ------------------------------------------------------------------
# Logs (SSE)
# ------------------------------------------------------------------

@router.get("/{instance_id}/logs")
async def stream_logs(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    tail: int = Query(settings.LOG_HISTORY_LINES, ge=1),
):
    """
    SSE endpoint — streams instance logs in real-time.

    curl:  curl -N -H "Authorization: Bearer <token>" http://localhost:8000/api/instances/{id}/logs
    """
    instance = await _get_user_instance(instance_id, current_user, db)
    logger.info("Log stream opened: id=%s user_id=%s", instance_id, current_user.id)

    if instance.miner_type == "docker":
        if not instance.container_id:
            raise HTTPException(409, "Instance has no running container")
        container_id = instance.container_id

        async def docker_event_generator():
            proc = await asyncio.create_subprocess_exec(
                "docker", "logs", "-f",
                "--timestamps",
                "--tail", str(tail),
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

        generator = docker_event_generator()
    else:
        if not instance.twitch_username:
            raise HTTPException(409, "Instance has no twitch_username configured")

        async def subprocess_event_generator():
            async for line in tail_log(instance_id, instance.twitch_username, history_lines=tail):
                yield f"data: {line.rstrip()}\n\n"

        generator = subprocess_event_generator()

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ------------------------------------------------------------------
# Subprocess-only: Streamers
# ------------------------------------------------------------------

@router.get("/{instance_id}/streamers")
async def get_streamers(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get the streamer list for a subprocess instance."""
    instance = await _get_user_instance(instance_id, current_user, db)
    if instance.miner_type != "subprocess":
        raise HTTPException(400, "Streamers are only available for subprocess instances")
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
    if instance.miner_type != "subprocess":
        raise HTTPException(400, "Streamers are only available for subprocess instances")
    config = _read_config(instance_id)
    config["streamers"] = data.streamers
    _write_config(instance_id, config)
    logger.info("Streamers updated: id=%s count=%d", instance_id, len(data.streamers))
    return {"streamers": data.streamers}


# ------------------------------------------------------------------
# Subprocess-only: Points Snapshot
# ------------------------------------------------------------------

@router.get("/points-snapshot", response_model=list[InstancePointsSnapshotResponse])
async def get_points_snapshot(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    history_lines: int = Query(2000, ge=100, le=20000),
    refresh: bool = Query(False),
):
    """Channel points snapshot for all subprocess instances."""
    if current_user.is_admin():
        result = await db.execute(
            select(MinerInstance).where(MinerInstance.miner_type == "subprocess")
        )
    else:
        result = await db.execute(
            select(MinerInstance).where(
                MinerInstance.user_id == current_user.id,
                MinerInstance.miner_type == "subprocess",
            )
        )

    snapshots: list[InstancePointsSnapshotResponse] = []
    for instance in result.scalars().all():
        if not instance.twitch_username:
            continue
        configured: list[str] = []
        try:
            config = _read_config(instance.id)
            configured = [str(s).strip() for s in config.get("streamers", []) if str(s).strip()]
        except Exception:
            pass

        points = get_instance_points_snapshot(
            instance.id,
            instance.twitch_username,
            history_lines=history_lines,
            refresh=refresh,
            expected_streamers=set(configured) if configured else None,
        )

        ordered: list[StreamerPointsSnapshot] = []
        seen: set[str] = set()
        for name in configured:
            key = name.lower()
            if key in points:
                ordered.append(StreamerPointsSnapshot(streamer=name, channel_points=points[key]))
                seen.add(key)
        for name, pts in sorted(points.items()):
            if name not in seen:
                ordered.append(StreamerPointsSnapshot(streamer=name, channel_points=pts))

        snapshots.append(InstancePointsSnapshotResponse(instance_id=instance.id, streamers=ordered))

    return snapshots
