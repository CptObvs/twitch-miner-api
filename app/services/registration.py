"""
Registration code service for secure registration.
"""

import secrets
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.database import RegistrationCode


def generate_random_code(length: int = 16) -> str:
    """Generate a random alphanumeric code."""
    return secrets.token_urlsafe(length)[:length].upper()


async def create_registration_code(
    db: AsyncSession,
    expires_in_hours: int = 24,
    created_by: str | None = None,
) -> str:
    """Create a new registration code."""
    code = generate_random_code()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)

    reg_code = RegistrationCode(
        code=code,
        expires_at=expires_at,
        created_by=created_by,
    )
    db.add(reg_code)
    await db.commit()

    return code


async def validate_registration_code(
    db: AsyncSession,
    code: str,
) -> bool:
    """Validate a registration code."""
    result = await db.execute(select(RegistrationCode).where(RegistrationCode.code == code))
    reg_code = result.scalar_one_or_none()

    if not reg_code or not reg_code.is_valid():
        return False

    return True


async def mark_code_as_used(
    db: AsyncSession,
    code: str,
    user_id: str,
) -> None:
    """Mark a registration code as used."""
    result = await db.execute(select(RegistrationCode).where(RegistrationCode.code == code))
    reg_code = result.scalar_one_or_none()

    if reg_code:
        reg_code.used_at = datetime.now(timezone.utc)
        reg_code.used_by = user_id
        db.add(reg_code)
        await db.commit()
