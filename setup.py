#!/usr/bin/env python3
"""
Setup script to initialize the database and create admin user.

Usage:
    python setup.py --create-admin
    python setup.py --username myotheruser --password mypass --role admin

Or run interactive mode:
    python setup.py
"""

import asyncio
import sys
import io

# Enable UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import argparse
from sqlalchemy import select
from app.models.database import init_db, User, async_session
from app.models.enums import UserRole
from app.services.auth import hash_password
from app.services.registration import create_registration_code


async def create_admin_user(username: str = "admin", password: str = "testpass123"):
    """Create the initial admin user."""
    async with async_session() as session:
        # Check if admin already exists
        result = await session.execute(select(User).where(User.username == username))
        if result.scalar_one_or_none():
            print(f"✓ Admin user '{username}' already exists")
            return False

        # Create admin user
        admin = User(
            username=username,
            password_hash=hash_password(password),
            role=UserRole.ADMIN,
        )
        session.add(admin)
        await session.commit()
        print(f"✓ Created admin user: {username}")
        print(f"  Password: {password}")
        print(f"  ⚠️  Please change the default password after first login!")
        return True


async def create_initial_codes(count: int = 5, expires_in_hours: int = 72):
    """Create initial registration codes."""
    async with async_session() as session:
        codes = []
        for i in range(count):
            code = await create_registration_code(session, expires_in_hours=expires_in_hours)
            codes.append(code)
            print(f"  Code {i+1}: {code}")
        return codes


async def list_users():
    """List all users in the database."""
    async with async_session() as session:
        result = await session.execute(select(User))
        users = result.scalars().all()

        if not users:
            print("No users in database")
            return

        print(f"\nFound {len(users)} user(s):")
        for user in users:
            role_icon = "👑" if user.is_admin() else "👤"
            print(f"  {role_icon} {user.username}: {user.role}")


async def main():
    parser = argparse.ArgumentParser(description="Setup Twitch Miner Backend")
    parser.add_argument(
        "--create-admin",
        action="store_true",
        help="Create default admin user (admin:testpass123)",
    )
    parser.add_argument(
        "--username",
        help="Username for new user",
    )
    parser.add_argument(
        "--password",
        help="Password for new user",
    )
    parser.add_argument(
        "--role",
        choices=["admin", "user"],
        default="user",
        help="Role for new user",
    )
    parser.add_argument(
        "--codes",
        type=int,
        help="Generate N registration codes",
    )
    parser.add_argument(
        "--codes-hours",
        type=int,
        default=72,
        help="Registration codes expire after N hours (default: 72)",
    )

    args = parser.parse_args()

    # Initialize database
    print("Initializing database...")
    await init_db()
    print("✓ Database initialized\n")

    # Create admin user
    if args.create_admin:
        await create_admin_user()

    # Create custom user
    if args.username and args.password:
        async with async_session() as session:
            result = await session.execute(select(User).where(User.username == args.username))
            if result.scalar_one_or_none():
                print(f"❌ User '{args.username}' already exists")
                return

            user = User(
                username=args.username,
                password_hash=hash_password(args.password),
                role=UserRole.ADMIN if args.role == "admin" else UserRole.USER,
            )
            session.add(user)
            await session.commit()
            role_icon = "👑" if args.role == "admin" else "👤"
            print(f"✓ Created user: {role_icon} {args.username} ({args.role})")

    # Generate registration codes
    if args.codes:
        print(f"\n✓ Generating {args.codes} registration code(s)...")
        codes = await create_initial_codes(args.codes, args.codes_hours)
        print(f"  Expiring in: {args.codes_hours} hours")

    # List users
    await list_users()


if __name__ == "__main__":
    asyncio.run(main())
