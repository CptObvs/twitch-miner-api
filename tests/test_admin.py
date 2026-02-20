"""Tests for admin endpoints."""

import pytest
from httpx import AsyncClient

from app.models.database import User
from tests.conftest import auth_header


# --- User Management ---


@pytest.mark.asyncio
async def test_admin_list_users(client: AsyncClient, admin_token: str, admin_user: User):
    """Admin should be able to list all users."""
    response = await client.get("/admin/users", headers=auth_header(admin_token))
    assert response.status_code == 200
    users = response.json()
    assert len(users) >= 1
    assert any(u["username"] == "admin" for u in users)


@pytest.mark.asyncio
async def test_user_cannot_list_users(client: AsyncClient, user_token: str, normal_user: User):
    """Normal user should not be able to list users."""
    response = await client.get("/admin/users", headers=auth_header(user_token))
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_admin_update_role(
    client: AsyncClient, admin_token: str, admin_user: User, normal_user: User
):
    """Admin should be able to change a user's role."""
    response = await client.patch(
        f"/admin/users/{normal_user.id}/role",
        json={"role": "admin"},
        headers=auth_header(admin_token),
    )
    assert response.status_code == 200
    assert response.json()["role"] == "admin"


@pytest.mark.asyncio
async def test_user_cannot_update_role(
    client: AsyncClient, user_token: str, normal_user: User, admin_user: User
):
    """Normal user should not be able to change roles."""
    response = await client.patch(
        f"/admin/users/{admin_user.id}/role",
        json={"role": "user"},
        headers=auth_header(user_token),
    )
    assert response.status_code == 403


# --- Invite Limit ---


@pytest.mark.asyncio
async def test_admin_update_invite_limit(
    client: AsyncClient, admin_token: str, admin_user: User, normal_user: User
):
    """Admin should be able to change a user's invite code limit."""
    response = await client.patch(
        f"/admin/users/{normal_user.id}/invite-limit",
        json={"max_invite_codes": 10},
        headers=auth_header(admin_token),
    )
    assert response.status_code == 200
    assert response.json()["max_invite_codes"] == 10


@pytest.mark.asyncio
async def test_user_cannot_update_invite_limit(
    client: AsyncClient, user_token: str, normal_user: User
):
    """Normal user should not be able to change invite limits."""
    response = await client.patch(
        f"/admin/users/{normal_user.id}/invite-limit",
        json={"max_invite_codes": 100},
        headers=auth_header(user_token),
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_negative_invite_limit_rejected(
    client: AsyncClient, admin_token: str, admin_user: User, normal_user: User
):
    """Negative invite limit should be rejected."""
    response = await client.patch(
        f"/admin/users/{normal_user.id}/invite-limit",
        json={"max_invite_codes": -1},
        headers=auth_header(admin_token),
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_invite_limit_affects_code_generation(
    client: AsyncClient,
    admin_token: str,
    user_token: str,
    admin_user: User,
    normal_user: User,
):
    """Increasing the invite limit should allow more code generation."""
    # Increase limit to 5
    await client.patch(
        f"/admin/users/{normal_user.id}/invite-limit",
        json={"max_invite_codes": 5},
        headers=auth_header(admin_token),
    )

    # User should be able to generate 5 codes
    for i in range(5):
        r = await client.post(
            "/codes/generate",
            json={"expires_in_hours": 24},
            headers=auth_header(user_token),
        )
        assert r.status_code == 200, f"Failed on code {i + 1}"

    # 6th should fail
    r = await client.post(
        "/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(user_token),
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_update_limit_nonexistent_user(
    client: AsyncClient, admin_token: str, admin_user: User
):
    """Updating invite limit for a non-existent user should return 404."""
    response = await client.patch(
        "/admin/users/nonexistent-id/invite-limit",
        json={"max_invite_codes": 5},
        headers=auth_header(admin_token),
    )
    assert response.status_code == 404


# --- Admin Code Endpoints ---


@pytest.mark.asyncio
async def test_admin_generate_code_via_admin_endpoint(
    client: AsyncClient, admin_token: str, admin_user: User
):
    """Admin code generation endpoint should still work."""
    response = await client.post(
        "/admin/codes/generate",
        json={"expires_in_hours": 48},
        headers=auth_header(admin_token),
    )
    assert response.status_code == 200
    assert "code" in response.json()


@pytest.mark.asyncio
async def test_user_cannot_use_admin_code_endpoint(
    client: AsyncClient, user_token: str, normal_user: User
):
    """Normal user should not access admin code endpoint."""
    response = await client.post(
        "/admin/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(user_token),
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_admin_list_all_codes(
    client: AsyncClient, admin_token: str, admin_user: User
):
    """Admin should see all codes via admin endpoint."""
    # Generate a code first
    await client.post(
        "/admin/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(admin_token),
    )

    response = await client.get("/admin/codes", headers=auth_header(admin_token))
    assert response.status_code == 200
    codes = response.json()
    assert len(codes) >= 1
    assert "created_by_username" in codes[0]
