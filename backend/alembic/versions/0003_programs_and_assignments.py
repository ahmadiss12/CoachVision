"""Programs, program days/exercises, and client assignments (Phase 2).

Revision ID: 0003_programs_and_assignments
Revises: 0002_roles_and_trainer_links
Create Date: 2026-07-14

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision: str = "0003_programs_and_assignments"
down_revision: Union[str, None] = "0002_roles_and_trainer_links"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _assignment_status() -> postgresql.ENUM:
    return postgresql.ENUM(
        "active", "completed", "cancelled", name="assignment_status", create_type=False
    )


def upgrade() -> None:
    bind = op.get_bind()
    online = not context.is_offline_mode()

    _assignment_status().create(bind, checkfirst=online)

    op.create_table(
        "programs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "trainer_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("weeks", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_programs_trainer_id", "programs", ["trainer_id"])

    op.create_table(
        "program_days",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "program_id",
            sa.Uuid(),
            sa.ForeignKey("programs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("day_index", sa.Integer(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.UniqueConstraint("program_id", "day_index", name="uq_program_days_program_day"),
    )

    op.create_table(
        "program_exercises",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "program_day_id",
            sa.Uuid(),
            sa.ForeignKey("program_days.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("exercise_id", sa.String(), sa.ForeignKey("exercises.id"), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("target_sets", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("target_reps", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("rest_seconds", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("external_load_kg", sa.Numeric(6, 2), nullable=True),
        sa.Column("notes", sa.String(300), nullable=True),
    )
    op.create_index("ix_program_exercises_day_id", "program_exercises", ["program_day_id"])

    op.create_table(
        "program_assignments",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "program_id",
            sa.Uuid(),
            sa.ForeignKey("programs.id", ondelete="CASCADE"),
            nullable=False,
        ),
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
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("status", _assignment_status(), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_program_assignments_client_id", "program_assignments", ["client_id"])
    op.create_index("ix_program_assignments_trainer_id", "program_assignments", ["trainer_id"])

    op.add_column(
        "sessions",
        sa.Column(
            "assignment_id",
            sa.Uuid(),
            sa.ForeignKey("program_assignments.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    bind = op.get_bind()
    online = not context.is_offline_mode()

    op.drop_column("sessions", "assignment_id")
    op.drop_index("ix_program_assignments_trainer_id", table_name="program_assignments")
    op.drop_index("ix_program_assignments_client_id", table_name="program_assignments")
    op.drop_table("program_assignments")
    op.drop_index("ix_program_exercises_day_id", table_name="program_exercises")
    op.drop_table("program_exercises")
    op.drop_table("program_days")
    op.drop_index("ix_programs_trainer_id", table_name="programs")
    op.drop_table("programs")

    _assignment_status().drop(bind, checkfirst=online)
