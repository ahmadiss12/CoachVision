"""Subscription plan on user accounts.

Revision ID: 0005_subscription_plan
Revises: 0004_chat_messages
Create Date: 2026-11-15

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision: str = "0005_subscription_plan"
down_revision: Union[str, None] = "0004_chat_messages"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _subscription_plan() -> postgresql.ENUM:
    return postgresql.ENUM("free", "plus", "pro", name="subscription_plan", create_type=False)


def upgrade() -> None:
    bind = op.get_bind()
    online = not context.is_offline_mode()

    _subscription_plan().create(bind, checkfirst=online)

    # server_default backfills every existing account as free.
    op.add_column(
        "users",
        sa.Column("subscription_plan", _subscription_plan(), nullable=False, server_default="free"),
    )


def downgrade() -> None:
    bind = op.get_bind()
    online = not context.is_offline_mode()

    op.drop_column("users", "subscription_plan")
    _subscription_plan().drop(bind, checkfirst=online)
