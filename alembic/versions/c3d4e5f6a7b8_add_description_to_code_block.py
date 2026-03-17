"""add description to code_block

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-03-17 18:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'code_block',
        sa.Column('description', sa.Text(), nullable=True,
                   comment='LLM-generated knowledge text, persisted before Milvus write'),
    )


def downgrade() -> None:
    op.drop_column('code_block', 'description')
