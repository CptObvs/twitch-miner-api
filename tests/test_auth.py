"""Tests for authentication endpoints."""

import pytest
from httpx import AsyncClient

from app.models.database import User
from app.services.registration import create_registration_code
from tests.conftest import auth_header


@pytest.mark.asyncio
async def test_register_with_valid_code(client: AsyncClient, db_session):
    """Registration with a valid code should succeed."""
    code = await create_registration_code(db_session, expires_in_hours=1)

    response = await client.post("/auth/register", json={
        "username": "newuser",
        "password": "securepass",
        "registration_code": code,
    })
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "newuser"
    assert "access_token" in data


@pytest.mark.asyncio
async def test_register_with_invalid_code(client: AsyncClient):
    """Registration with an invalid code should fail."""
    response = await client.post("/auth/register", json={
        "username": "newuser",
        "password": "securepass",
        "registration_code": "INVALID-CODE",
    })
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_register_duplicate_username(client: AsyncClient, db_session, normal_user: User):
    """Registration with an existing username should fail."""
    code = await create_registration_code(db_session, expires_in_hours=1)

    response = await client.post("/auth/register", json={
        "username": "testuser",
        "password": "anotherpass",
        "registration_code": code,
    })
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient, normal_user: User):
    """Login with correct credentials should return a token."""
    response = await client.post("/auth/token", data={
        "username": "testuser",
        "password": "userpass",
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["username"] == "testuser"


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient, normal_user: User):
    """Login with wrong password should fail."""
    response = await client.post("/auth/token", data={
        "username": "testuser",
        "password": "wrongpass",
    })
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_me(client: AsyncClient, user_token: str, normal_user: User):
    """Getting current user info should work with a valid token."""
    response = await client.get("/auth/me", headers=auth_header(user_token))
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "testuser"
    assert data["role"] == "user"


@pytest.mark.asyncio
async def test_get_me_no_auth(client: AsyncClient):
    """Getting current user without a token should fail."""
    response = await client.get("/auth/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_change_password(client: AsyncClient, user_token: str, normal_user: User):
    """Changing password with correct current password should succeed."""
    response = await client.post("/auth/change-password", json={
        "current_password": "userpass",
        "new_password": "newpass123",
    }, headers=auth_header(user_token))
    assert response.status_code == 200

    # Login with the new password
    response = await client.post("/auth/token", data={
        "username": "testuser",
        "password": "newpass123",
    })
    assert response.status_code == 200
