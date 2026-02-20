import asyncio
import logging
import re
import shutil
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.database import User, MinerInstance, get_db
from app.models.enums import InstanceState
from app.models.schemas import InstanceCreate, InstanceResponse, InstanceStatus
from app.services.auth import get_current_user
from app.services.miner_manager import miner_manager

router = APIRouter(prefix="/instances", tags=["instances"])
logger = logging.getLogger("uvicorn.error")

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mGKH]")


def _instance_to_response(instance: MinerInstance) -> InstanceResponse:
    ui_url = None
    if instance.status == InstanceState.RUNNING:
        ui_url = f"/instances/{instance.id}/ui"
    return InstanceResponse(
        id=instance.id,
        user_id=instance.user_id,
        status=instance.status,
        container_id=instance.container_id,
        port=instance.port,
        ui_url=ui_url,
        created_at=instance.created_at,
        last_started_at=instance.last_started_at,
        last_stopped_at=instance.last_stopped_at,
    )


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
        count = len(result.scalars().all())
        if count >= settings.MAX_INSTANCES_PER_USER:
            raise HTTPException(
                400,
                f"Maximum {settings.MAX_INSTANCES_PER_USER} instances per user",
            )

    instance = MinerInstance(user_id=current_user.id)
    db.add(instance)
    await db.commit()
    await db.refresh(instance)
    logger.info("Instance created: id=%s user_id=%s", instance.id, current_user.id)
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
    """Stop container (if running) then delete DB record."""
    instance = await _get_user_instance(instance_id, current_user, db)

    if instance.status != InstanceState.STOPPED:
        await miner_manager.stop(instance_id, db_session=db)

    await db.delete(instance)
    await db.commit()

    instance_dir = settings.INSTANCES_DIR / instance_id
    shutil.rmtree(instance_dir, ignore_errors=True)

    logger.info("Instance deleted: id=%s user_id=%s", instance_id, current_user.id)


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
        running = result.scalars().all()
        if len(running) >= settings.MAX_INSTANCES_PER_USER:
            raise HTTPException(
                400,
                f"Maximum {settings.MAX_INSTANCES_PER_USER} running instances per user",
            )

    try:
        container_id = await miner_manager.start(instance_id, db_session=db)
    except RuntimeError as e:
        raise HTTPException(409, str(e))

    await db.refresh(instance)
    logger.info("Instance started: id=%s container=%s", instance_id, container_id[:12])
    return InstanceStatus(
        id=instance_id,
        status=InstanceState.RUNNING,
        container_id=container_id,
        port=instance.port,
    )


@router.post("/{instance_id}/stop", response_model=InstanceStatus)
async def stop_instance(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Stop the Docker container. Blocks until fully stopped."""
    await _get_user_instance(instance_id, current_user, db)
    logger.info("Stop requested: id=%s user_id=%s", instance_id, current_user.id)
    await miner_manager.stop(instance_id, db_session=db)
    return InstanceStatus(id=instance_id, status=InstanceState.STOPPED)


@router.get("/{instance_id}/status", response_model=InstanceStatus)
async def instance_status(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Live-check container status against Docker, then return DB state."""
    await miner_manager.reconcile_instance_status(instance_id, db_session=db)
    instance = await _get_user_instance(instance_id, current_user, db)
    return InstanceStatus(
        id=instance_id,
        status=instance.status,
        container_id=instance.container_id,
        port=instance.port,
    )


@router.get("/{instance_id}/logs")
async def stream_logs(
    instance_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    tail: int = 100,
):
    """
    SSE endpoint — streams Docker container logs in real-time.

    curl:  curl -N -H "Authorization: Bearer <token>" http://localhost:8000/api/instances/{id}/logs
    """
    instance = await _get_user_instance(instance_id, current_user, db)
    if not instance.container_id:
        raise HTTPException(409, "Instance has no running container")

    container_id = instance.container_id
    logger.info("Log stream opened: id=%s user_id=%s", instance_id, current_user.id)

    async def event_generator():
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
                if not line:
                    continue
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
