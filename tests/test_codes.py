"""Tests for invite code endpoints (user-facing and admin)."""

import pytest
from httpx import AsyncClient

from app.models.database import User, RegistrationCode
from app.services.registration import create_registration_code
from tests.conftest import auth_header


# --- User Code Generation ---


@pytest.mark.asyncio
async def test_user_generate_code(client: AsyncClient, user_token: str, normal_user: User):
    """Normal user should be able to generate an invite code."""
    response = await client.post(
        "/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(user_token),
    )
    assert response.status_code == 200
    data = response.json()
    assert "code" in data
    assert "expires_at" in data


@pytest.mark.asyncio
async def test_user_code_limit(client: AsyncClient, user_token: str, normal_user: User):
    """Normal user should be blocked after reaching their code limit (default 2)."""
    # Generate first code
    r1 = await client.post(
        "/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(user_token),
    )
    assert r1.status_code == 200

    # Generate second code
    r2 = await client.post(
        "/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(user_token),
    )
    assert r2.status_code == 200

    # Third code should be blocked
    r3 = await client.post(
        "/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(user_token),
    )
    assert r3.status_code == 403
    assert "limit" in r3.json()["detail"].lower()


@pytest.mark.asyncio
async def test_admin_no_code_limit(client: AsyncClient, admin_token: str, admin_user: User):
    """Admin users should have no limit on code generation."""
    for _ in range(5):
        r = await client.post(
            "/codes/generate",
            json={"expires_in_hours": 24},
            headers=auth_header(admin_token),
        )
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_unauthenticated_generate_code(client: AsyncClient):
    """Unauthenticated requests should be rejected."""
    response = await client.post("/codes/generate", json={"expires_in_hours": 24})
    assert response.status_code == 401


# --- Code Listing ---


@pytest.mark.asyncio
async def test_user_list_own_codes(client: AsyncClient, user_token: str, normal_user: User):
    """Normal user should only see their own codes."""
    # Generate a code
    await client.post(
        "/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(user_token),
    )

    response = await client.get("/codes/", headers=auth_header(user_token))
    assert response.status_code == 200
    codes = response.json()
    assert len(codes) == 1
    assert "id" in codes[0]
    assert "is_valid" in codes[0]


@pytest.mark.asyncio
async def test_admin_list_all_codes(
    client: AsyncClient,
    admin_token: str,
    user_token: str,
    admin_user: User,
    normal_user: User,
):
    """Admin should see all codes from all users."""
    # User generates a code
    await client.post(
        "/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(user_token),
    )
    # Admin generates a code
    await client.post(
        "/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(admin_token),
    )

    response = await client.get("/codes/", headers=auth_header(admin_token))
    assert response.status_code == 200
    codes = response.json()
    assert len(codes) >= 2


# --- Code Deletion ---


@pytest.mark.asyncio
async def test_user_delete_own_code(client: AsyncClient, user_token: str, normal_user: User):
    """User should be able to delete their own unused code."""
    # Generate a code
    gen_resp = await client.post(
        "/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(user_token),
    )
    assert gen_resp.status_code == 200

    # List codes to get the ID
    list_resp = await client.get("/codes/", headers=auth_header(user_token))
    code_id = list_resp.json()[0]["id"]

    # Delete it
    del_resp = await client.delete(f"/codes/{code_id}", headers=auth_header(user_token))
    assert del_resp.status_code == 204

    # Verify it's gone
    list_resp2 = await client.get("/codes/", headers=auth_header(user_token))
    assert len(list_resp2.json()) == 0


@pytest.mark.asyncio
async def test_user_cannot_delete_others_code(
    client: AsyncClient,
    user_token: str,
    admin_token: str,
    normal_user: User,
    admin_user: User,
):
    """Normal user should not be able to delete another user's code."""
    # Admin generates a code
    await client.post(
        "/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(admin_token),
    )

    # Admin lists codes to get ID
    list_resp = await client.get("/codes/", headers=auth_header(admin_token))
    code_id = list_resp.json()[0]["id"]

    # User tries to delete it
    del_resp = await client.delete(f"/codes/{code_id}", headers=auth_header(user_token))
    assert del_resp.status_code == 403


@pytest.mark.asyncio
async def test_cannot_delete_used_code(
    client: AsyncClient, admin_token: str, admin_user: User, db_session
):
    """Used codes should not be deletable."""
    # Create a code and mark it as used
    code_str = await create_registration_code(db_session, created_by=admin_user.id)
    from sqlalchemy import select
    result = await db_session.execute(
        select(RegistrationCode).where(RegistrationCode.code == code_str)
    )
    reg_code = result.scalar_one()
    from datetime import datetime, timezone
    reg_code.used_at = datetime.now(timezone.utc)
    reg_code.used_by = admin_user.id
    await db_session.commit()

    # Try to delete it
    del_resp = await client.delete(
        f"/codes/{reg_code.id}", headers=auth_header(admin_token)
    )
    assert del_resp.status_code == 400
    assert "already been used" in del_resp.json()["detail"]


@pytest.mark.asyncio
async def test_delete_nonexistent_code(client: AsyncClient, admin_token: str, admin_user: User):
    """Deleting a non-existent code should return 404."""
    del_resp = await client.delete(
        "/codes/nonexistent-id", headers=auth_header(admin_token)
    )
    assert del_resp.status_code == 404


@pytest.mark.asyncio
async def test_code_limit_freed_after_delete(
    client: AsyncClient, user_token: str, normal_user: User
):
    """After deleting a code, user should be able to generate a new one."""
    # Fill up the limit
    for _ in range(2):
        await client.post(
            "/codes/generate",
            json={"expires_in_hours": 24},
            headers=auth_header(user_token),
        )

    # Should be blocked
    r = await client.post(
        "/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(user_token),
    )
    assert r.status_code == 403

    # Delete one code
    list_resp = await client.get("/codes/", headers=auth_header(user_token))
    code_id = list_resp.json()[0]["id"]
    await client.delete(f"/codes/{code_id}", headers=auth_header(user_token))

    # Should be able to generate again
    r = await client.post(
        "/codes/generate",
        json={"expires_in_hours": 24},
        headers=auth_header(user_token),
    )
    assert r.status_code == 200
