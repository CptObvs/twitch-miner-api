"""Retain the 1,000 most recently active tracked IPs and index activity queries."""

from alembic import op

revision = "0006_limit_ip_tracking"
down_revision = "0005_connected_ips"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Historical tracker writes used ISO 'T', while ORM writes use a space.
    # Normalize the separator so the index orders both formats chronologically.
    op.execute("UPDATE connected_ips SET last_seen = REPLACE(last_seen, 'T', ' '), first_seen = REPLACE(first_seen, 'T', ' ')")
    op.create_index("ix_connected_ips_last_seen_ip", "connected_ips", ["last_seen", "ip_address"])
    op.execute("""
        DELETE FROM connected_ips WHERE ip_address NOT IN (
            SELECT ip_address FROM connected_ips
            ORDER BY last_seen DESC, ip_address DESC LIMIT 1000
        )
    """)
    op.create_index("ix_banned_ips_banned_until", "banned_ips", ["banned_until"])


def downgrade() -> None:
    op.drop_index("ix_banned_ips_banned_until", table_name="banned_ips")
    op.drop_index("ix_connected_ips_last_seen_ip", table_name="connected_ips")
    # Pruned historical IP records cannot be recovered by a downgrade.
