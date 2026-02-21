"""Add miner_type and subprocess fields to miner_instances

Revision ID: 0002_add_miner_type
Revises: 0001_init
Create Date: 2026-02-21 00:00:00.000000
"""

from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "0002_add_miner_type"
down_revision: Union[str, Sequence[str], None] = "0001_init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "miner_instances",
        sa.Column("miner_type", sa.String(), nullable=False, server_default="docker"),
    )
    op.add_column(
        "miner_instances",
        sa.Column("twitch_username", sa.String(), nullable=True),
    )
    op.add_column(
        "miner_instances",
        sa.Column("pid", sa.Integer(), nullable=True),
    )
    op.add_column(
        "miner_instances",
        sa.Column("enable_analytics", sa.Boolean(), nullable=False, server_default="0"),
    )
    op.add_column(
        "miner_instances",
        sa.Column("analytics_port", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("miner_instances", "analytics_port")
    op.drop_column("miner_instances", "enable_analytics")
    op.drop_column("miner_instances", "pid")
    op.drop_column("miner_instances", "twitch_username")
    op.drop_column("miner_instances", "miner_type")
