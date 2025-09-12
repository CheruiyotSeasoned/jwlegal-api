"""Add Google OAuth fields to users

Revision ID: c659d3341a2d
Revises: 7aedb2f9e8ca
Create Date: 2025-09-03 06:24:02.861469

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c659d3341a2d'
down_revision: Union[str, None] = '7aedb2f9e8ca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column("users", sa.Column("auth_provider", sa.String(length=50), nullable=True, server_default="local"))
    op.add_column("users", sa.Column("google_id", sa.String(length=50), nullable=True))
    op.add_column("users", sa.Column("google_refresh_token", sa.Text(), nullable=True))
    op.create_index("ix_users_google_id", "users", ["google_id"], unique=True)


def downgrade():
    op.drop_index("ix_users_google_id", table_name="users")
    op.drop_column("users", "google_refresh_token")
    op.drop_column("users", "google_id")
    op.drop_column("users", "auth_provider")
