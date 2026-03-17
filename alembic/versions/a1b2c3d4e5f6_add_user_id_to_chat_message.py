"""add user_id to chat_message

Revision ID: a1b2c3d4e5f6
Revises: f0d67a5be465
Create Date: 2026-03-17 10:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'f0d67a5be465'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('chat_message', sa.Column('user_id', sa.String(100), nullable=True, comment='Feishu user_id (employee number)'))


def downgrade() -> None:
    op.drop_column('chat_message', 'user_id')
