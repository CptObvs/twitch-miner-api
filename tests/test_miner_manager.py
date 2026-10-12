"""Container upgrades must reach API instances and preserve their data mount."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.models.enums import InstanceState, MinerType
from app.services import miner_manager as module


@pytest.fixture
def manager(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "INSTANCES_DIR", tmp_path)
    monkeypatch.setattr(settings, "DOCKER_IMAGE", "rangermix/twitch-drops-miner:latest")
    monkeypatch.setattr(settings, "DOCKER_PULL_ON_START", True)
    manager = module.DockerContainerManager()
    manager._reserve_instance_start = AsyncMock(return_value=True)
    manager._update_instance = AsyncMock()
    return manager


async def test_drops_pulls_before_recreate_and_enables_browser(manager, monkeypatch, tmp_path):
    docker = AsyncMock(side_effect=[(0, "pulled", ""), (0, "removed", ""), (0, "new-id", "")])
    monkeypatch.setattr(module, "_run_docker_cmd", docker)
    data = tmp_path / "one" / "data"
    data.mkdir(parents=True)
    (data / "cookies.jar").write_bytes(b"existing session")

    assert await manager.start("one") == "new-id"
    calls = docker.call_args_list
    assert calls[0].args == ("pull", settings.DOCKER_IMAGE)
    assert calls[1].args == ("rm", "-f", "miner-one")
    run = list(calls[2].args)
    assert run[-1] == settings.DOCKER_IMAGE
    assert "--init" in run
    assert run[run.index("--shm-size") + 1] == "256m"
    assert run[run.index("-e") + 1] == f"TZ={settings.DOCKER_TIMEZONE}"
    assert f"{data.resolve()}:/app/data" in run
    assert (data / "cookies.jar").read_bytes() == b"existing session"


@pytest.mark.parametrize("result", [(1, "", "registry unavailable"), asyncio.TimeoutError(), OSError("no docker")])
async def test_failed_pull_leaves_previous_container_and_resets_start(manager, monkeypatch, result):
    docker = AsyncMock(side_effect=[result])
    monkeypatch.setattr(module, "_run_docker_cmd", docker)
    with pytest.raises(RuntimeError):
        await manager.start("one")
    assert docker.await_count == 1
    assert manager.get_port("one") is None
    assert manager._update_instance.call_args.kwargs["status"] == InstanceState.STOPPED


async def test_local_image_can_skip_pull(manager, monkeypatch):
    monkeypatch.setattr(settings, "DOCKER_PULL_ON_START", False)
    docker = AsyncMock(side_effect=[(0, "", ""), (0, "container", "")])
    monkeypatch.setattr(module, "_run_docker_cmd", docker)
    await manager.start("one")
    assert all(call.args[0] != "pull" for call in docker.call_args_list)


async def test_v2_keeps_its_existing_start_command(manager, monkeypatch):
    docker = AsyncMock(side_effect=[(0, "", ""), (0, "container", "")])
    monkeypatch.setattr(module, "_run_docker_cmd", docker)
    await manager.start("one", miner_type=MinerType.TwitchPointsMinerV2)
    run = docker.call_args.args
    assert run[-1] == settings.V2_DOCKER_IMAGE
    assert "--shm-size" not in run
    assert docker.call_args_list[0].args[0] == "rm"


async def test_failed_run_releases_port_and_clears_container_fields(manager, monkeypatch):
    docker = AsyncMock(side_effect=[(0, "", ""), (0, "", ""), (1, "", "run failed")])
    monkeypatch.setattr(module, "_run_docker_cmd", docker)
    with pytest.raises(RuntimeError, match="run failed"):
        await manager.start("one")
    assert manager.get_port("one") is None
    update = manager._update_instance.call_args.kwargs
    assert update["status"] == InstanceState.STOPPED
    assert update["container_id"] is None and update["port"] is None
