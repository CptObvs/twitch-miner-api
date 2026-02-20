"""add analytics fields

Revision ID: b3c4d5e6f7g8
Revises: a1b2c3d4e5f6
Create Date: 2026-02-20 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3c4d5e6f7g8'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite requires batch mode for ALTER TABLE
    with op.batch_alter_table("miner_instances") as batch_op:
        batch_op.add_column(
            sa.Column("enable_analytics", sa.Boolean(), nullable=False, server_default="0")
        )
        batch_op.add_column(
            sa.Column("analytics_port", sa.Integer(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("miner_instances") as batch_op:
        batch_op.drop_column("analytics_port")
        batch_op.drop_column("enable_analytics")
