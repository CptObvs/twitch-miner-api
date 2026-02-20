"""
Test fixtures for the Twitch Miner Backend API.

Uses an in-memory SQLite database for test isolation.
"""

from typing import AsyncGenerator

import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.models.database import Base, get_db, User
from app.models.enums import UserRole
from app.services.auth import hash_password, create_access_token


# In-memory SQLite for tests
TEST_DATABASE_URL = "sqlite+aiosqlite://"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create tables, yield a session, then drop tables."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client with the test DB injected."""
    from main import app

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def admin_user(db_session: AsyncSession) -> User:
    """Create an admin user in the test DB."""
    user = User(
        username="admin",
        password_hash=hash_password("adminpass"),
        role=UserRole.ADMIN,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def normal_user(db_session: AsyncSession) -> User:
    """Create a normal user in the test DB."""
    user = User(
        username="testuser",
        password_hash=hash_password("userpass"),
        role=UserRole.USER,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def admin_token(admin_user: User) -> str:
    """Get a JWT token for the admin user."""
    return create_access_token(admin_user.id, admin_user.username)


@pytest_asyncio.fixture
async def user_token(normal_user: User) -> str:
    """Get a JWT token for a normal user."""
    return create_access_token(normal_user.id, normal_user.username)


def auth_header(token: str) -> dict[str, str]:
    """Build an Authorization header."""
    return {"Authorization": f"Bearer {token}"}
