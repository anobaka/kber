"""add retry_count to code_block

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-17 15:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'code_block',
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0',
                   comment='Number of times this block has been retried'),
    )


def downgrade() -> None:
    op.drop_column('code_block', 'retry_count')
