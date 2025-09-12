"""Add Google OAuth fields to users

Revision ID: 851d5cb1f4e7
Revises: c659d3341a2d
Create Date: 2025-09-03 06:25:03.432148

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "851d5cb1f4e7"
down_revision: Union[str, None] = "c659d3341a2d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
