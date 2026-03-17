"""add response_feedback table

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-03-17 20:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'response_feedback',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('chat_id', sa.String(100), nullable=False),
        sa.Column('message_id', sa.String(100), nullable=False),
        sa.Column('user_open_id', sa.String(100), nullable=True),
        sa.Column('question', sa.Text(), nullable=True),
        sa.Column('answer', sa.Text(), nullable=True),
        sa.Column(
            'rating',
            sa.Enum('helpful', 'not_helpful', name='feedback_rating_enum'),
            nullable=False,
        ),
        sa.Column('reason', sa.Text(), nullable=True, comment='Optional reason when rated not_helpful'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('message_id'),
    )
    op.create_index('idx_feedback_chat', 'response_feedback', ['chat_id'])
    op.create_index('idx_feedback_rating', 'response_feedback', ['rating'])


def downgrade() -> None:
    op.drop_index('idx_feedback_rating', table_name='response_feedback')
    op.drop_index('idx_feedback_chat', table_name='response_feedback')
    op.drop_table('response_feedback')
