"""
Manages Docker container lifecycles for Twitch miner instances.

Each instance maps to one Docker container:
  - Started via: docker run -d -p 127.0.0.1:{port}:8080 -v {data_dir}:/app/data {image}
  - Stopped via: docker stop {container_id} && docker rm {container_id}
  - Status via:  docker inspect --format={{.State.Running}} {container_id}
"""

import asyncio
import logging
import socket
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.database import MinerInstance, async_session
from app.models.enums import InstanceState

logger = logging.getLogger(__name__)

# Sentinel object to distinguish "don't update this field" from "set to None"
_UNSET = object()


async def _run_docker_cmd(*args: str, timeout: int = 30) -> tuple[int, str, str]:
    """
    Run a docker CLI command asynchronously.
    Returns (returncode, stdout, stderr).
    """
    proc = await asyncio.create_subprocess_exec(
        "docker", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return (
        proc.returncode,
        stdout_bytes.decode("utf-8", errors="replace").strip(),
        stderr_bytes.decode("utf-8", errors="replace").strip(),
    )


def _find_free_port(start: int) -> int:
    """Find the next free TCP port starting from start."""
    port = start
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1


class DockerContainerManager:
    """Singleton that manages Docker containers for all miner instances."""

    def __init__(self):
        self._allocated_ports: dict[str, int] = {}
        self._next_port: int = settings.DOCKER_PORT_BASE

    # ------------------------------------------------------------------
    # Port allocation
    # ------------------------------------------------------------------

    def _allocate_port(self, instance_id: str) -> int:
        if instance_id in self._allocated_ports:
            return self._allocated_ports[instance_id]
        port = _find_free_port(self._next_port)
        self._allocated_ports[instance_id] = port
        self._next_port = port + 1
        return port

    def _release_port(self, instance_id: str) -> None:
        self._allocated_ports.pop(instance_id, None)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    async def _update_instance(
        self,
        instance_id: str,
        *,
        status: InstanceState,
        container_id=_UNSET,
        port=_UNSET,
        set_last_started_at: bool = False,
        set_last_stopped_at: bool = False,
        db_session: AsyncSession | None = None,
    ) -> None:
        values: dict = {"status": status}
        if container_id is not _UNSET:
            values["container_id"] = container_id
        if port is not _UNSET:
            values["port"] = port
        if set_last_started_at:
            values["last_started_at"] = datetime.now(timezone.utc)
        if set_last_stopped_at:
            values["last_stopped_at"] = datetime.now(timezone.utc)

        stmt = update(MinerInstance).where(MinerInstance.id == instance_id).values(**values)

        if db_session is not None:
            await db_session.execute(stmt)
            await db_session.commit()
        else:
            async with async_session() as db:
                await db.execute(stmt)
                await db.commit()

    async def _reserve_instance_start(
        self, instance_id: str, db_session: AsyncSession | None = None
    ) -> bool:
        """
        Atomically flip STOPPED -> RUNNING only if currently STOPPED.
        Returns True if the reservation succeeded.
        """
        stmt = (
            update(MinerInstance)
            .where(
                MinerInstance.id == instance_id,
                MinerInstance.status == InstanceState.STOPPED,
            )
            .values(
                status=InstanceState.RUNNING,
                last_started_at=datetime.now(timezone.utc),
            )
        )
        if db_session is not None:
            result = await db_session.execute(stmt)
            await db_session.commit()
            return (result.rowcount or 0) == 1

        async with async_session() as db:
            result = await db.execute(stmt)
            await db.commit()
            return (result.rowcount or 0) == 1

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        instance_id: str,
        db_session: AsyncSession | None = None,
    ) -> str:
        """
        Start a Docker container for the given instance.
        Returns the container ID.
        Raises RuntimeError if already running or Docker fails.
        """
        reserved = await self._reserve_instance_start(instance_id, db_session=db_session)
        if not reserved:
            raise RuntimeError(f"Instance {instance_id} is already running or transitioning")

        port = self._allocate_port(instance_id)
        data_dir = (settings.INSTANCES_DIR / instance_id / "data").resolve()
        data_dir.mkdir(parents=True, exist_ok=True)

        container_name = f"miner-{instance_id}"

        # Remove any stale container with the same name (e.g. after a crash)
        await _run_docker_cmd("rm", "-f", container_name, timeout=10)

        cmd = [
            "run", "-d",
            "--name", container_name,
            "-p", f"127.0.0.1:{port}:8080",
            "-v", f"{data_dir}:/app/data",
            settings.DOCKER_IMAGE,
        ]

        try:
            returncode, stdout, stderr = await _run_docker_cmd(*cmd, timeout=60)
        except asyncio.TimeoutError:
            self._release_port(instance_id)
            await self._update_instance(
                instance_id, status=InstanceState.STOPPED, db_session=db_session
            )
            raise RuntimeError(f"docker run timed out for instance {instance_id}")

        if returncode != 0:
            self._release_port(instance_id)
            await self._update_instance(
                instance_id, status=InstanceState.STOPPED, db_session=db_session
            )
            raise RuntimeError(f"docker run failed: {stderr}")

        container_id = stdout  # docker run -d prints the full container ID
        await self._update_instance(
            instance_id,
            status=InstanceState.RUNNING,
            container_id=container_id,
            port=port,
            db_session=db_session,
        )
        logger.info(
            "Started container %s for instance %s on port %d",
            container_id[:12], instance_id, port,
        )
        return container_id

    async def stop(
        self,
        instance_id: str,
        db_session: AsyncSession | None = None,
    ) -> bool:
        """
        Stop and remove the Docker container.
        Returns True if stopped, False if wasn't running.
        """
        async def _fetch_instance(db: AsyncSession) -> MinerInstance | None:
            result = await db.execute(
                select(MinerInstance).where(MinerInstance.id == instance_id)
            )
            return result.scalar_one_or_none()

        if db_session is not None:
            inst = await _fetch_instance(db_session)
        else:
            async with async_session() as db:
                inst = await _fetch_instance(db)

        if not inst or inst.status == InstanceState.STOPPED:
            return False

        container_id = inst.container_id

        # Set STOPPING so other requests see the transition
        await self._update_instance(
            instance_id, status=InstanceState.STOPPING, db_session=db_session
        )

        if container_id:
            stop_code, _, stop_err = await _run_docker_cmd(
                "stop", "-t", str(settings.DOCKER_STOP_TIMEOUT), container_id,
                timeout=settings.DOCKER_STOP_TIMEOUT + 15,
            )
            if stop_code != 0:
                logger.warning("docker stop non-zero for %s: %s", container_id[:12], stop_err)

            rm_code, _, rm_err = await _run_docker_cmd("rm", container_id, timeout=15)
            if rm_code != 0:
                logger.warning("docker rm non-zero for %s: %s", container_id[:12], rm_err)

        self._release_port(instance_id)
        await self._update_instance(
            instance_id,
            status=InstanceState.STOPPED,
            container_id=None,
            port=None,
            set_last_stopped_at=True,
            db_session=db_session,
        )
        logger.info(
            "Stopped container %s for instance %s",
            (container_id[:12] if container_id else "N/A"), instance_id,
        )
        return True

    # ------------------------------------------------------------------
    # Status reconciliation
    # ------------------------------------------------------------------

    async def _container_is_running(self, container_id: str) -> bool:
        """Check via docker inspect if a container is in 'running' state."""
        if not container_id:
            return False
        try:
            rc, stdout, _ = await _run_docker_cmd(
                "inspect", "--format", "{{.State.Running}}", container_id,
                timeout=10,
            )
            return rc == 0 and stdout.strip() == "true"
        except asyncio.TimeoutError:
            return False

    async def reconcile_instance_status(
        self, instance_id: str, db_session: AsyncSession | None = None
    ) -> None:
        """Check if the recorded container is still running; fix DB if not."""
        async def _reconcile(db: AsyncSession) -> None:
            result = await db.execute(
                select(MinerInstance).where(MinerInstance.id == instance_id)
            )
            inst = result.scalar_one_or_none()
            if not inst or inst.status == InstanceState.STOPPED:
                return
            running = await self._container_is_running(inst.container_id or "")
            if not running:
                inst.status = InstanceState.STOPPED
                inst.container_id = None
                inst.port = None
                inst.last_stopped_at = datetime.now(timezone.utc)
                await db.commit()
                self._release_port(instance_id)

        if db_session is not None:
            await _reconcile(db_session)
        else:
            async with async_session() as db:
                await _reconcile(db)

    async def reconcile_all_on_startup(self) -> None:
        """
        On startup: rebuild port map from DB, then verify each running container.
        Flips stale instances to STOPPED.
        """
        async with async_session() as db:
            result = await db.execute(select(MinerInstance))
            all_instances = result.scalars().all()

        # Rebuild in-memory port map from DB
        for inst in all_instances:
            if inst.status != InstanceState.STOPPED and inst.port is not None:
                self._allocated_ports[inst.id] = inst.port
                if inst.port >= self._next_port:
                    self._next_port = inst.port + 1

        non_stopped = [i for i in all_instances if i.status != InstanceState.STOPPED]
        if not non_stopped:
            return

        changed = 0
        async with async_session() as db:
            for inst in non_stopped:
                running = await self._container_is_running(inst.container_id or "")
                if not running:
                    live = (await db.execute(
                        select(MinerInstance).where(MinerInstance.id == inst.id)
                    )).scalar_one_or_none()
                    if live:
                        live.status = InstanceState.STOPPED
                        live.container_id = None
                        live.port = None
                        live.last_stopped_at = datetime.now(timezone.utc)
                        self._release_port(inst.id)
                        changed += 1
            if changed:
                await db.commit()
                logger.info("Reconciled %d stale instance(s) to STOPPED on startup", changed)

    async def cleanup_orphan_containers(self) -> None:
        """Periodic task: re-check all running instances against Docker."""
        await self.reconcile_all_on_startup()

    async def shutdown_all(self) -> None:
        """Stop all running containers on app shutdown."""
        async with async_session() as db:
            result = await db.execute(
                select(MinerInstance).where(MinerInstance.status != InstanceState.STOPPED)
            )
            instances = result.scalars().all()

        for inst in instances:
            try:
                await self.stop(inst.id)
            except Exception as e:
                logger.error("Error stopping instance %s on shutdown: %s", inst.id, e)

    def get_port(self, instance_id: str) -> int | None:
        """Return the in-memory allocated port for an instance, if known."""
        return self._allocated_ports.get(instance_id)


# Global singleton
miner_manager = DockerContainerManager()
