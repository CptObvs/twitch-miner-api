"""
Admin router for managing users and registration codes.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import User, RegistrationCode, get_db
from app.models.schemas import (
    UserResponse,
    UpdateUserRoleRequest,
    UpdateInviteLimitRequest,
    GenerateCodeRequest,
    RegistrationCodeResponse,
    RegistrationCodeDetailResponse,
)
from app.services.auth import get_current_user
from app.services.registration import create_registration_code

router = APIRouter(prefix="/admin", tags=["Admin"])


# --- User Management ---


@router.get("/users", response_model=list[UserResponse])
async def list_all_users(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[UserResponse]:
    """
    List all users in the system.

    Only ADMIN users can access this endpoint.
    """
    if not current_user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can list all users",
        )

    result = await db.execute(select(User))
    users = result.scalars().all()
    return users


@router.patch("/users/{user_id}/role", response_model=UserResponse)
async def update_user_role(
    user_id: str,
    data: UpdateUserRoleRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    """
    Update a user's role.

    Only ADMIN users can access this endpoint.
    Allows changing a user's role between ADMIN and USER.
    """
    if not current_user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can update user roles",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found",
        )

    user.role = data.role
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user


@router.patch("/users/{user_id}/invite-limit", response_model=UserResponse)
async def update_invite_limit(
    user_id: str,
    data: UpdateInviteLimitRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    """
    Update a user's invite code limit.

    Only ADMIN users can access this endpoint.
    Sets how many active (unused, non-expired) invite codes a user can have.
    """
    if not current_user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can update invite limits",
        )

    if data.max_invite_codes < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invite limit cannot be negative",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found",
        )

    user.max_invite_codes = data.max_invite_codes
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user


# --- Registration Codes (admin-only legacy endpoints) ---


@router.post("/codes/generate", response_model=RegistrationCodeResponse)
async def generate_registration_code(
    request: GenerateCodeRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RegistrationCodeResponse:
    """
    Generate a new registration code (admin only).

    Prefer using POST /codes/generate which is available to all users.
    The code will expire after the specified number of hours (default: 24).
    """
    if not current_user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can use this endpoint. Use POST /codes/generate instead.",
        )

    code = await create_registration_code(
        db, expires_in_hours=request.expires_in_hours, created_by=current_user.id
    )

    result = await db.execute(select(RegistrationCode).where(RegistrationCode.code == code))
    reg_code = result.scalar_one_or_none()

    return RegistrationCodeResponse(
        code=reg_code.code,
        created_at=reg_code.created_at,
        expires_at=reg_code.expires_at,
    )


@router.get("/codes", response_model=list[RegistrationCodeDetailResponse])
async def list_registration_codes(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[RegistrationCodeDetailResponse]:
    """
    List all registration codes (used and unused).

    Only ADMIN users can access this endpoint.
    Prefer using GET /codes which shows user-specific codes.
    """
    if not current_user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can use this endpoint. Use GET /codes instead.",
        )

    result = await db.execute(select(RegistrationCode))
    codes = result.scalars().all()

    # Resolve user IDs to usernames
    user_ids = set()
    for code in codes:
        if code.used_by:
            user_ids.add(code.used_by)
        if code.created_by:
            user_ids.add(code.created_by)

    username_map = {}
    if user_ids:
        user_result = await db.execute(select(User).where(User.id.in_(user_ids)))
        username_map = {u.id: u.username for u in user_result.scalars().all()}

    return [
        RegistrationCodeDetailResponse(
            id=code.id,
            code=code.code,
            created_at=code.created_at,
            expires_at=code.expires_at,
            used_at=code.used_at,
            used_by=username_map.get(code.used_by) if code.used_by else None,
            created_by_username=username_map.get(code.created_by) if code.created_by else None,
            is_valid=code.is_valid(),
        )
        for code in codes
    ]
