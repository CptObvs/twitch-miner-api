"""Initial schema

Revision ID: 0001_init
Revises:
Create Date: 2026-02-20 00:00:00.000000
"""

from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "0001_init"
down_revision: Union[str, Sequence[str], None] = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("role", sa.Enum("ADMIN", "USER", name="userrole"), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("max_invite_codes", sa.Integer(), nullable=False, server_default="2"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )

    op.create_table(
        "registration_codes",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("code", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("used_at", sa.DateTime(), nullable=True),
        sa.Column("used_by", sa.String(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["used_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code"),
    )
    op.create_index("ix_registration_codes_code", "registration_codes", ["code"], unique=True)

    op.create_table(
        "miner_instances",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("STOPPED", "RUNNING", "STOPPING", name="instancestate"),
            nullable=False,
            server_default="stopped",
        ),
        sa.Column("container_id", sa.String(), nullable=True),
        sa.Column("port", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_started_at", sa.DateTime(), nullable=True),
        sa.Column("last_stopped_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("miner_instances")
    op.drop_index("ix_registration_codes_code", table_name="registration_codes")
    op.drop_table("registration_codes")
    op.drop_table("users")
