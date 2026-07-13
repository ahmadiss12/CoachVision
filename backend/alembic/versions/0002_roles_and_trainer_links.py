"""Roles, trainer-client links, and client invites (Phase 1 multi-tenancy).

Revision ID: 0002_roles_and_trainer_links
Revises: 0001_baseline
Create Date: 2026-07-13

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision: str = "0002_roles_and_trainer_links"
down_revision: Union[str, None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _user_role() -> postgresql.ENUM:
    return postgresql.ENUM("client", "trainer", "admin", name="user_role", create_type=False)


def _trainer_client_status() -> postgresql.ENUM:
    return postgresql.ENUM("active", "ended", name="trainer_client_status", create_type=False)


def _invite_status() -> postgresql.ENUM:
    return postgresql.ENUM(
        "pending", "accepted", "revoked", "expired", name="invite_status", create_type=False
    )


def upgrade() -> None:
    bind = op.get_bind()
    online = not context.is_offline_mode()

    _user_role().create(bind, checkfirst=online)
    _trainer_client_status().create(bind, checkfirst=online)
    _invite_status().create(bind, checkfirst=online)

    op.add_column(
        "users",
        sa.Column("role", _user_role(), nullable=False, server_default="client"),
    )

    op.create_table(
        "trainer_clients",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "trainer_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "client_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", _trainer_client_status(), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("trainer_id", "client_id", name="uq_trainer_clients_pair"),
    )
    op.create_index("ix_trainer_clients_trainer_id", "trainer_clients", ["trainer_id"])
    op.create_index("ix_trainer_clients_client_id", "trainer_clients", ["client_id"])

    op.create_table(
        "client_invites",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "trainer_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token", sa.String(64), nullable=False, unique=True),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("status", _invite_status(), nullable=False, server_default="pending"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "accepted_by",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_client_invites_trainer_id", "client_invites", ["trainer_id"])


def downgrade() -> None:
    bind = op.get_bind()
    online = not context.is_offline_mode()

    op.drop_index("ix_client_invites_trainer_id", table_name="client_invites")
    op.drop_table("client_invites")
    op.drop_index("ix_trainer_clients_client_id", table_name="trainer_clients")
    op.drop_index("ix_trainer_clients_trainer_id", table_name="trainer_clients")
    op.drop_table("trainer_clients")
    op.drop_column("users", "role")

    _invite_status().drop(bind, checkfirst=online)
    _trainer_client_status().drop(bind, checkfirst=online)
    _user_role().drop(bind, checkfirst=online)
