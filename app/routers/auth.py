"""
Authentication router with OAuth2 Password Flow.
"""

from collections import defaultdict
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import User, get_db
from app.models.schemas import (
    RegisterRequest,
    TokenResponse,
    UserResponse,
    ChangePasswordRequest,
)
from app.services.auth import create_access_token, get_current_user, hash_password, verify_password
from app.services.registration import (
    validate_registration_code,
    mark_code_as_used,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ------------------------------------------------------------------
# Rate limiter
# ------------------------------------------------------------------

_RATE_LIMITER_MAX_IPS = 10_000


class _RateLimiter:
    """Simple in-memory rate limiter: max `limit` requests per `window` seconds per IP."""

    def __init__(self, limit: int = 10, window: int = 60, message: str = "Too many requests. Try again in a minute."):
        self.limit = limit
        self.window = window
        self.message = message
        # IP -> list of timestamps
        self._attempts: dict[str, list[float]] = defaultdict(list)

    def check(self, ip: str) -> None:
        now = time.monotonic()
        # Evict oldest IP when dict grows too large (prevents memory leak under attack)
        if len(self._attempts) > _RATE_LIMITER_MAX_IPS and ip not in self._attempts:
            oldest = next(iter(self._attempts))
            del self._attempts[oldest]
        # Remove expired entries for this IP
        self._attempts[ip] = [t for t in self._attempts[ip] if now - t < self.window]
        if len(self._attempts[ip]) >= self.limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=self.message,
            )
        self._attempts[ip].append(now)


_login_limiter = _RateLimiter(limit=10, window=60, message="Too many login attempts. Try again in a minute.")
_register_limiter = _RateLimiter(limit=10, window=60, message="Too many registration attempts. Try again in a minute.")


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: Request,
    data: RegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """
    Register a new user account.

    Requires a valid registration code to prevent unauthorized registrations.
    Returns an access token immediately upon successful registration.
    """
    _register_limiter.check(request.client.host)

    # Validate registration code
    if not await validate_registration_code(db, data.registration_code):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired registration code",
        )

    # Check if username already exists
    result = await db.execute(select(User).where(User.username == data.username))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{data.username}' is already taken",
        )

    user = User(
        username=data.username,
        password_hash=hash_password(data.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Mark registration code as used
    await mark_code_as_used(db, data.registration_code, user.id)

    token = create_access_token(user.id, user.username)
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        username=user.username,
    )


@router.post("/token", response_model=TokenResponse)
async def login(
    request: Request,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """
    OAuth2 compatible token endpoint.

    Use this endpoint with the 'Authorize' button in SwaggerUI:
    1. Click 'Authorize'
    2. Enter your username and password
    3. All protected endpoints will be automatically authenticated

    This endpoint follows the OAuth2 Password Flow specification.
    """
    _login_limiter.check(request.client.host)

    result = await db.execute(select(User).where(User.username == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user.id, user.username)
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        username=user.username,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    """
    Get the currently authenticated user's information.

    Requires a valid Bearer token in the Authorization header.
    """
    return current_user


@router.post("/change-password", response_model=dict)
async def change_password(
    data: ChangePasswordRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """
    Change the current user's password.

    Requires:
    - Valid Bearer token
    - Current password (for verification)
    - New password (must be different from current)
    """
    # Verify current password
    if not verify_password(data.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )

    # Hash new password
    new_password_hash = hash_password(data.new_password)

    # Update user in database
    current_user.password_hash = new_password_hash
    db.add(current_user)
    await db.commit()

    return {"message": "Password changed successfully"}