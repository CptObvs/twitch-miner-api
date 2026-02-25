"""
Manages Docker container lifecycles for all miner instance types.
  Drops: docker run -d -p 127.0.0.1:{port}:8080 -v {data_dir}:/app/data {DOCKER_IMAGE}
  V2:    docker run -d -v {run_py}:/usr/src/app/run.py:ro -v {cookies}:/usr/src/app/cookies {V2_DOCKER_IMAGE}
"""

import asyncio
import logging
import sys
import socket
import textwrap
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.database import MinerInstance, async_session
from app.models.enums import InstanceState, MinerType

logger = logging.getLogger(__name__)

_UNSET = object()


async def _run_docker_cmd(*args: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run a docker CLI command. Returns (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "docker", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode, out.decode(errors="replace").strip(), err.decode(errors="replace").strip()


class DockerContainerManager:
    """Singleton that manages Docker containers for all miner instances."""

    def __init__(self):
        self._allocated_ports: dict[str, int] = {}
        self._next_port: int = settings.DOCKER_PORT_BASE

    def _allocate_port(self, instance_id: str) -> int:
        if instance_id not in self._allocated_ports:
            port = self._next_port
            while True:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex(("127.0.0.1", port)) != 0:
                        break
                port += 1
            self._allocated_ports[instance_id] = port
            self._next_port = port + 1
        return self._allocated_ports[instance_id]

    def _release_port(self, instance_id: str) -> None:
        self._allocated_ports.pop(instance_id, None)

    @asynccontextmanager
    async def _use_db(self, db_session: AsyncSession | None):
        """Yield the provided session, or open a fresh one if None."""
        if db_session is not None:
            yield db_session
        else:
            async with async_session() as db:
                yield db

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
        async with self._use_db(db_session) as db:
            await db.execute(update(MinerInstance).where(MinerInstance.id == instance_id).values(**values))
            await db.commit()

    async def _reserve_instance_start(self, instance_id: str, db_session: AsyncSession | None = None) -> bool:
        """Atomically flip STOPPED -> RUNNING. Returns True if succeeded."""
        stmt = (
            update(MinerInstance)
            .where(MinerInstance.id == instance_id, MinerInstance.status == InstanceState.STOPPED)
            .values(status=InstanceState.RUNNING, last_started_at=datetime.now(timezone.utc))
        )
        async with self._use_db(db_session) as db:
            result = await db.execute(stmt)
            await db.commit()
            return (result.rowcount or 0) == 1

    def _mark_stopped(self, inst: MinerInstance) -> None:
        """Set stopped-state fields on an ORM object (caller must commit)."""
        inst.status = InstanceState.STOPPED
        inst.container_id = None
        inst.port = None
        inst.last_stopped_at = datetime.now(timezone.utc)
        self._release_port(inst.id)

    def _generate_v2_run_script(self, instance_id: str, twitch_username: str, streamers: list[str]) -> Path:
        """Generate run.py for a V2 instance and return its path."""
        instance_dir = settings.INSTANCES_DIR / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)

        if streamers:
            streamer_lines = "\n".join(
                f'                    "{s.strip()}",' for s in streamers if s.strip()
            )
            streamers_block = f"\n{streamer_lines}\n                "
        else:
            streamers_block = ""

        script = textwrap.dedent(f"""\
            # -*- coding: utf-8 -*-

            import logging
            import sys
            from TwitchChannelPointsMiner import TwitchChannelPointsMiner
            from TwitchChannelPointsMiner.logger import LoggerSettings
            from TwitchChannelPointsMiner.classes.Chat import ChatPresence
            from TwitchChannelPointsMiner.classes.Settings import Priority, FollowersOrder
            from TwitchChannelPointsMiner.classes.entities.Bet import (
                Strategy, BetSettings, Condition, OutcomeKeys, FilterCondition, DelayMode,
            )
            from TwitchChannelPointsMiner.classes.entities.Streamer import StreamerSettings

            # Redirect stdout and stderr to output.log file
            log_file = open("/app/output.log", "a", encoding="utf-8")
            sys.stdout = log_file
            sys.stderr = log_file

            twitch_miner = TwitchChannelPointsMiner(
                username="{twitch_username}",
                claim_drops_startup=False,
                priority=[
                    Priority.STREAK,
                    Priority.DROPS,
                    Priority.ORDER,
                ],
                enable_analytics=False,
                logger_settings=LoggerSettings(
                    save=False,
                    console_level=logging.INFO,
                    console_username=False,
                    auto_clear=True,
                    time_zone="",
                    file_level=logging.INFO,
                    emoji=True,
                    less=True,
                    colored=True,
                ),
                streamer_settings=StreamerSettings(
                    make_predictions=True,
                    follow_raid=True,
                    claim_drops=True,
                    claim_moments=True,
                    watch_streak=True,
                    community_goals=False,
                    chat=ChatPresence.ONLINE,
                    bet=BetSettings(
                        strategy=Strategy.SMART,
                        percentage=5,
                        percentage_gap=20,
                        max_points=50000,
                        stealth_mode=True,
                        delay_mode=DelayMode.FROM_END,
                        delay=6,
                        minimum_points=20000,
                        filter_condition=FilterCondition(
                            by=OutcomeKeys.TOTAL_USERS,
                            where=Condition.LTE,
                            value=800,
                        ),
                    ),
                ),
            )

            twitch_miner.mine(
                [{streamers_block}],
                followers=False,
                followers_order=FollowersOrder.ASC,
            )
            """)
        run_py = instance_dir / "run.py"
        run_py.write_text(script, encoding="utf-8")
        return run_py

    async def get_recent_logs(self, container_id: str, tail: int = 20, instance_id: str | None = None, instance_type: MinerType | None = None) -> list[str]:
        """Fetch the last `tail` log lines. V2 reads from output.log, Drops from docker logs."""
        if instance_type == MinerType.TwitchPointsMinerV2 and instance_id:
            log_file = settings.INSTANCES_DIR / instance_id / "output.log"
            if not log_file.exists():
                return []
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                return [line.rstrip('\n') for line in lines[-tail:] if line.strip()]
            except Exception:
                return []
        try:
            rc, stdout, _ = await _run_docker_cmd("logs", "--tail", str(tail), container_id, timeout=10)
        except asyncio.TimeoutError:
            return []
        return stdout.splitlines() if rc == 0 and stdout else []

    async def start(
        self,
        instance_id: str,
        *,
        miner_type: MinerType = MinerType.TwitchDropsMiner,
        twitch_username: str | None = None,
        streamers: list[str] | None = None,
        db_session: AsyncSession | None = None,
    ) -> str:
        """Start a Docker container. Returns container ID. Raises RuntimeError on failure."""
        if not await self._reserve_instance_start(instance_id, db_session=db_session):
            raise RuntimeError(f"Instance {instance_id} is already running or transitioning")

        container_name = f"miner-{instance_id}"
        await _run_docker_cmd("rm", "-f", container_name, timeout=10)

        if miner_type == MinerType.TwitchPointsMinerV2:
            run_py = self._generate_v2_run_script(instance_id, twitch_username or "", streamers or [])
            cookies_dir = (settings.INSTANCES_DIR / instance_id / "cookies").resolve()
            cookies_dir.mkdir(parents=True, exist_ok=True)
            output_log = (settings.INSTANCES_DIR / instance_id / "output.log").resolve()
            output_log.parent.mkdir(parents=True, exist_ok=True)
            output_log.touch(exist_ok=True)  # ensure file exists so Docker mounts a file, not a dir
            cmd = [
                "run", "-d", "--name", container_name,
                "-v", f"{run_py.resolve()}:/usr/src/app/run.py:ro",
                "-v", f"{cookies_dir}:/usr/src/app/cookies",
                "-v", f"{output_log}:/app/output.log",
                settings.V2_DOCKER_IMAGE,
            ]
            port = None
        else:
            port = self._allocate_port(instance_id)
            data_dir = (settings.INSTANCES_DIR / instance_id / "data").resolve()
            data_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "run", "-d", "--name", container_name,
                "-p", f"127.0.0.1:{port}:8080",
                "-v", f"{data_dir}:/app/data",
                settings.DOCKER_IMAGE,
            ]

        try:
            returncode, stdout, stderr = await _run_docker_cmd(*cmd, timeout=60)
        except asyncio.TimeoutError:
            returncode, stderr = -1, f"docker run timed out for instance {instance_id}"

        if returncode != 0:
            if port is not None:
                self._release_port(instance_id)
            await self._update_instance(instance_id, status=InstanceState.STOPPED, db_session=db_session)
            raise RuntimeError(stderr)

        container_id = stdout
        update_kw: dict = dict(status=InstanceState.RUNNING, container_id=container_id, db_session=db_session)
        if port is not None:
            update_kw["port"] = port
        await self._update_instance(instance_id, **update_kw)

        if port is not None:
            logger.info("Started Drops container %s for instance %s on port %d", container_id[:12], instance_id, port)
        else:
            logger.info("Started V2 container %s for instance %s", container_id[:12], instance_id)
        return container_id

    async def stop(self, instance_id: str, db_session: AsyncSession | None = None) -> bool:
        """Stop and remove the container. Port and DB always cleaned up, even on Docker failure."""
        async with self._use_db(db_session) as db:
            inst = (await db.execute(
                select(MinerInstance).where(MinerInstance.id == instance_id)
            )).scalar_one_or_none()

        if not inst or inst.status == InstanceState.STOPPED:
            return False

        container_id = inst.container_id
        await self._update_instance(instance_id, status=InstanceState.STOPPING, db_session=db_session)

        if container_id:
            try:
                rc, _, err = await _run_docker_cmd(
                    "stop", "-t", str(settings.DOCKER_STOP_TIMEOUT), container_id,
                    timeout=settings.DOCKER_STOP_TIMEOUT + 15,
                )
                if rc != 0:
                    logger.warning("docker stop non-zero for %s: %s", container_id[:12], err)
                rc, _, err = await _run_docker_cmd("rm", container_id, timeout=15)
                if rc != 0:
                    logger.warning("docker rm non-zero for %s: %s", container_id[:12], err)
            except asyncio.TimeoutError:
                logger.warning("docker stop/rm timed out for %s — forcing local cleanup", container_id[:12])

        # Always run, regardless of Docker outcome
        self._release_port(instance_id)
        await self._update_instance(
            instance_id, status=InstanceState.STOPPED,
            container_id=None, port=None, set_last_stopped_at=True, db_session=db_session,
        )
        logger.info("Stopped container %s for instance %s", container_id[:12] if container_id else "N/A", instance_id)
        return True

    async def _container_is_running(self, container_id: str) -> bool:
        if not container_id:
            return False
        try:
            rc, stdout, _ = await _run_docker_cmd(
                "inspect", "--format", "{{.State.Running}}", container_id, timeout=10,
            )
            return rc == 0 and stdout.strip() == "true"
        except asyncio.TimeoutError:
            return False

    async def reconcile_instance_status(self, instance_id: str, db_session: AsyncSession | None = None) -> None:
        """Check if the recorded container is still running; fix DB if not."""
        async with self._use_db(db_session) as db:
            inst = (await db.execute(
                select(MinerInstance).where(MinerInstance.id == instance_id)
            )).scalar_one_or_none()
            if not inst or inst.status == InstanceState.STOPPED:
                return
            if not await self._container_is_running(inst.container_id or ""):
                self._mark_stopped(inst)
                await db.commit()

    async def reconcile_all_on_startup(self) -> None:
        """Rebuild port map and verify all running containers on startup."""
        async with self._use_db(None) as db:
            # Only load non-stopped instances — no need to inspect already-stopped ones
            non_stopped = (await db.execute(
                select(MinerInstance).where(MinerInstance.status != InstanceState.STOPPED)
            )).scalars().all()

            if not non_stopped:
                return

            for inst in non_stopped:
                if inst.miner_type == MinerType.TwitchDropsMiner and inst.port is not None:
                    self._allocated_ports[inst.id] = inst.port
                    if inst.port >= self._next_port:
                        self._next_port = inst.port + 1

            changed = 0
            for inst in non_stopped:
                if not await self._container_is_running(inst.container_id or ""):
                    self._mark_stopped(inst)
                    changed += 1

            if changed:
                await db.commit()
                logger.info("Reconciled %d stale instance(s) to STOPPED on startup", changed)

    async def shutdown_all(self) -> None:
        """Stop all running containers on app shutdown (in parallel)."""
        async with async_session() as db:
            instances = (await db.execute(
                select(MinerInstance).where(MinerInstance.status != InstanceState.STOPPED)
            )).scalars().all()

        if not instances:
            return

        logger.info("Stopping %d running instance(s) ...", len(instances))

        async def _stop_safe(inst: MinerInstance) -> None:
            try:
                await self.stop(inst.id)
            except Exception as e:
                logger.error("Error stopping instance %s on shutdown: %s", inst.id, e)

        await asyncio.gather(*(_stop_safe(inst) for inst in instances))

    def get_port(self, instance_id: str) -> int | None:
        return self._allocated_ports.get(instance_id)


# Global singleton
miner_manager = DockerContainerManager()