"""
User-facing router for managing invite/registration codes.

All authenticated users can generate, list, and delete their own codes.
Normal users are limited by their max_invite_codes setting.
Admins have no limit and can see/delete all codes.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import User, RegistrationCode, get_db
from app.models.schemas import (
    GenerateCodeRequest,
    RegistrationCodeResponse,
    RegistrationCodeDetailResponse,
)
from app.services.auth import get_current_user
from app.services.registration import create_registration_code

router = APIRouter(prefix="/codes", tags=["Invite Codes"])


@router.post("/generate", response_model=RegistrationCodeResponse)
async def generate_code(
    request: GenerateCodeRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RegistrationCodeResponse:
    """
    Generate a new invite code.

    All authenticated users can generate codes.
    Non-admin users are limited to their `max_invite_codes` active codes.
    """
    if not current_user.is_admin():
        # Count active (unused + not expired) codes created by this user
        result = await db.execute(
            select(func.count()).select_from(RegistrationCode).where(
                RegistrationCode.created_by == current_user.id,
                RegistrationCode.used_at.is_(None),
            )
        )
        active_count = result.scalar()

        # Filter out expired codes in Python (since is_valid checks tz)
        all_codes_result = await db.execute(
            select(RegistrationCode).where(
                RegistrationCode.created_by == current_user.id,
                RegistrationCode.used_at.is_(None),
            )
        )
        active_count = sum(1 for c in all_codes_result.scalars().all() if c.is_valid())

        if active_count >= current_user.max_invite_codes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invite code limit reached ({current_user.max_invite_codes}). "
                       "Delete unused codes or ask an admin to increase your limit.",
            )

    code = await create_registration_code(
        db,
        expires_in_hours=request.expires_in_hours,
        created_by=current_user.id,
    )

    result = await db.execute(
        select(RegistrationCode).where(RegistrationCode.code == code)
    )
    reg_code = result.scalar_one_or_none()

    return RegistrationCodeResponse(
        code=reg_code.code,
        created_at=reg_code.created_at,
        expires_at=reg_code.expires_at,
    )


@router.get("/", response_model=list[RegistrationCodeDetailResponse])
async def list_codes(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[RegistrationCodeDetailResponse]:
    """
    List invite codes.

    - Regular users: see only their own codes
    - Admins: see all codes
    """
    if current_user.is_admin():
        result = await db.execute(select(RegistrationCode))
    else:
        result = await db.execute(
            select(RegistrationCode).where(
                RegistrationCode.created_by == current_user.id
            )
        )
    codes = result.scalars().all()

    # Resolve user IDs to usernames for used_by and created_by
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


@router.delete("/{code_id}", status_code=204)
async def delete_code(
    code_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Delete an unused invite code.

    - Regular users can only delete their own unused codes.
    - Admins can delete any unused code.
    - Used codes cannot be deleted (history preservation).
    """
    result = await db.execute(
        select(RegistrationCode).where(RegistrationCode.id == code_id)
    )
    reg_code = result.scalar_one_or_none()

    if not reg_code:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Code not found",
        )

    # Non-admins can only delete their own codes
    if not current_user.is_admin() and reg_code.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own codes",
        )

    # Prevent deletion of used codes
    if reg_code.used_at is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a code that has already been used",
        )

    await db.delete(reg_code)
    await db.commit()
