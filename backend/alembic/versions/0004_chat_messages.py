"""Trainer-client chat messages (Phase 4, step 2).

Revision ID: 0004_chat_messages
Revises: 0003_programs_and_assignments
Create Date: 2026-07-18

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_chat_messages"
down_revision: Union[str, None] = "0003_programs_and_assignments"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "sender_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "recipient_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_chat_messages_pair_time",
        "chat_messages",
        ["sender_id", "recipient_id", "created_at"],
    )
    op.create_index(
        "ix_chat_messages_unread",
        "chat_messages",
        ["recipient_id", "read_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_chat_messages_unread", table_name="chat_messages")
    op.drop_index("ix_chat_messages_pair_time", table_name="chat_messages")
    op.drop_table("chat_messages")
