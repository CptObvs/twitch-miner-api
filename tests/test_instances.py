"""Tests for instance CRUD and lifecycle endpoints (Docker-based backend)."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import User, MinerInstance
from app.models.enums import InstanceState
from tests.conftest import auth_header


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

async def _create_instance(client: AsyncClient, token: str) -> dict:
    """Create a blank instance and return the response JSON."""
    resp = await client.post("/instances/", json={}, headers=auth_header(token))
    assert resp.status_code == 201
    return resp.json()


# ------------------------------------------------------------------
# CRUD tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_instance(client: AsyncClient, user_token: str, normal_user: User):
    """User should be able to create a blank miner instance."""
    data = await _create_instance(client, user_token)
    assert "id" in data
    assert data["status"] == "stopped"
    assert data["container_id"] is None
    assert data["port"] is None


@pytest.mark.asyncio
async def test_instance_limit_for_user(client: AsyncClient, user_token: str, normal_user: User):
    """Normal user should be limited to MAX_INSTANCES_PER_USER instances."""
    for _ in range(2):
        await _create_instance(client, user_token)

    r = await client.post("/instances/", json={}, headers=auth_header(user_token))
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_list_instances(client: AsyncClient, user_token: str, normal_user: User):
    """User should see their own instances."""
    await _create_instance(client, user_token)

    response = await client.get("/instances/", headers=auth_header(user_token))
    assert response.status_code == 200
    assert len(response.json()) == 1


@pytest.mark.asyncio
async def test_get_instance(client: AsyncClient, user_token: str, normal_user: User):
    """User should be able to get a specific instance."""
    instance_id = (await _create_instance(client, user_token))["id"]

    response = await client.get(f"/instances/{instance_id}", headers=auth_header(user_token))
    assert response.status_code == 200
    assert response.json()["id"] == instance_id


@pytest.mark.asyncio
async def test_delete_instance(client: AsyncClient, user_token: str, normal_user: User):
    """User should be able to delete their instance."""
    instance_id = (await _create_instance(client, user_token))["id"]

    del_resp = await client.delete(f"/instances/{instance_id}", headers=auth_header(user_token))
    assert del_resp.status_code == 204

    get_resp = await client.get(f"/instances/{instance_id}", headers=auth_header(user_token))
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_user_cannot_see_other_users_instance(
    client: AsyncClient,
    user_token: str,
    admin_token: str,
    normal_user: User,
    admin_user: User,
):
    """Normal user should not see instances of other users."""
    instance_id = (await _create_instance(client, admin_token))["id"]

    get_resp = await client.get(f"/instances/{instance_id}", headers=auth_header(user_token))
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_admin_can_see_all_instances(
    client: AsyncClient,
    user_token: str,
    admin_token: str,
    normal_user: User,
    admin_user: User,
):
    """Admin should see instances from all users."""
    await _create_instance(client, user_token)

    response = await client.get("/instances/", headers=auth_header(admin_token))
    assert response.status_code == 200
    assert len(response.json()) >= 1


# ------------------------------------------------------------------
# Status lifecycle tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_new_instance_has_stopped_status(client: AsyncClient, user_token: str, normal_user: User):
    """A newly created instance should have status 'stopped'."""
    data = await _create_instance(client, user_token)
    assert data["status"] == "stopped"
    assert "is_running" not in data


@pytest.mark.asyncio
async def test_status_endpoint_returns_status_field(
    client: AsyncClient, user_token: str, normal_user: User
):
    """GET /status should return the status enum field."""
    instance_id = (await _create_instance(client, user_token))["id"]

    # reconcile_instance_status returns early for STOPPED instances — no docker call needed
    status_resp = await client.get(
        f"/instances/{instance_id}/status", headers=auth_header(user_token)
    )
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["status"] == "stopped"
    assert data["id"] == instance_id
    assert "is_running" not in data


@pytest.mark.asyncio
async def test_stop_already_stopped_returns_200(
    client: AsyncClient, user_token: str, normal_user: User
):
    """Stopping an already-stopped instance should return 200 (idempotent)."""
    instance_id = (await _create_instance(client, user_token))["id"]

    stop_resp = await client.post(
        f"/instances/{instance_id}/stop", headers=auth_header(user_token)
    )
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "stopped"


@pytest.mark.asyncio
async def test_stop_sets_stopping_then_stopped(
    client: AsyncClient,
    user_token: str,
    normal_user: User,
    db_session: AsyncSession,
):
    """
    Verify the stop flow: the DB should transition through
    RUNNING -> STOPPING -> STOPPED.
    """
    instance_id = (await _create_instance(client, user_token))["id"]

    # Manually set instance to RUNNING in DB
    result = await db_session.execute(
        select(MinerInstance).where(MinerInstance.id == instance_id)
    )
    inst = result.scalar_one()
    inst.status = InstanceState.RUNNING
    inst.container_id = "abc123"
    await db_session.commit()

    observed_states: list[str] = []

    async def mock_stop(iid: str, db_session=None) -> bool:
        if iid != instance_id:
            return False
        from sqlalchemy import select as sa_select
        res = await db_session.execute(
            sa_select(MinerInstance).where(MinerInstance.id == iid)
        )
        mi = res.scalar_one_or_none()
        if mi:
            mi.status = InstanceState.STOPPING
            await db_session.commit()
            await db_session.refresh(mi)
            observed_states.append(mi.status.value)

            mi.status = InstanceState.STOPPED
            mi.container_id = None
            await db_session.commit()
        return True

    with patch("app.routers.instances.miner_manager") as mock_mgr:
        mock_mgr.stop = AsyncMock(side_effect=mock_stop)

        stop_resp = await client.post(
            f"/instances/{instance_id}/stop", headers=auth_header(user_token)
        )

    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "stopped"
    assert "stopping" in observed_states


@pytest.mark.asyncio
async def test_status_shows_stopping_during_shutdown(
    client: AsyncClient,
    user_token: str,
    normal_user: User,
    db_session: AsyncSession,
):
    """
    When the instance is in STOPPING state (mid-shutdown),
    the /status endpoint should report 'stopping'.
    We mock reconcile_instance_status to prevent docker calls.
    """
    instance_id = (await _create_instance(client, user_token))["id"]

    # Manually set to STOPPING in DB
    result = await db_session.execute(
        select(MinerInstance).where(MinerInstance.id == instance_id)
    )
    inst = result.scalar_one()
    inst.status = InstanceState.STOPPING
    await db_session.commit()

    with patch("app.routers.instances.miner_manager") as mock_mgr:
        mock_mgr.reconcile_instance_status = AsyncMock(return_value=None)

        status_resp = await client.get(
            f"/instances/{instance_id}/status", headers=auth_header(user_token)
        )
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "stopping"

    # List and detail endpoints also reflect STOPPING (no reconcile call there)
    get_resp = await client.get(f"/instances/{instance_id}", headers=auth_header(user_token))
    assert get_resp.json()["status"] == "stopping"

    list_resp = await client.get("/instances/", headers=auth_header(user_token))
    instance_data = [i for i in list_resp.json() if i["id"] == instance_id]
    assert len(instance_data) == 1
    assert instance_data[0]["status"] == "stopping"


@pytest.mark.asyncio
async def test_start_not_possible_when_already_running(
    client: AsyncClient,
    user_token: str,
    normal_user: User,
    db_session: AsyncSession,
):
    """Starting an already-running instance should return 409."""
    instance_id = (await _create_instance(client, user_token))["id"]

    # Set status to RUNNING in DB to simulate a running container
    result = await db_session.execute(
        select(MinerInstance).where(MinerInstance.id == instance_id)
    )
    inst = result.scalar_one()
    inst.status = InstanceState.RUNNING
    inst.container_id = "abc123"
    await db_session.commit()

    start_resp = await client.post(
        f"/instances/{instance_id}/start", headers=auth_header(user_token)
    )
    assert start_resp.status_code == 409


@pytest.mark.asyncio
async def test_instance_response_contains_status_field(
    client: AsyncClient, user_token: str, normal_user: User
):
    """All instance responses should use 'status' enum, not 'is_running' bool."""
    instance_id = (await _create_instance(client, user_token))["id"]

    get_resp = await client.get(f"/instances/{instance_id}", headers=auth_header(user_token))
    data = get_resp.json()
    assert "status" in data
    assert data["status"] in ("stopped", "running", "stopping")
    assert "is_running" not in data

    list_resp = await client.get("/instances/", headers=auth_header(user_token))
    for inst in list_resp.json():
        assert "status" in inst
        assert inst["status"] in ("stopped", "running", "stopping")
        assert "is_running" not in inst
