"""Baseline: freeze the pre-Alembic schema (13 tables).

Existing databases created by the old ``Base.metadata.create_all`` bootstrap
already have this schema, so ``upgrade`` detects that case (via the ``users``
table) and only records the revision without touching anything.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-07-12

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _difficulty_level() -> postgresql.ENUM:
    return postgresql.ENUM(
        "beginner", "intermediate", "advanced", name="difficulty_level", create_type=False
    )


def _session_status() -> postgresql.ENUM:
    return postgresql.ENUM(
        "planned", "active", "completed", "aborted", name="session_status", create_type=False
    )


def _fatigue_level() -> postgresql.ENUM:
    return postgresql.ENUM("low", "moderate", "high", name="fatigue_level", create_type=False)


def upgrade() -> None:
    bind = op.get_bind()
    online = not context.is_offline_mode()
    if online and sa.inspect(bind).has_table("users"):
        # Pre-Alembic database: schema already exists, just adopt it.
        return

    _difficulty_level().create(bind, checkfirst=online)
    _session_status().create(bind, checkfirst=online)
    _fatigue_level().create(bind, checkfirst=online)

    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("email", sa.String(), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("avatar_url", sa.String(), nullable=True),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
        sa.Column("height_cm", sa.Numeric(5, 2), nullable=True),
        sa.Column("weight_kg", sa.Numeric(6, 2), nullable=True),
        sa.Column("body_fat_percent", sa.Numeric(5, 2), nullable=True),
        sa.Column("timezone", sa.String(), nullable=False),
        sa.Column("locale", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "body_metrics",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("entry_date", sa.Date(), nullable=False),
        sa.Column("weight_kg", sa.Numeric(6, 2), nullable=False),
        sa.Column("body_fat_percent", sa.Numeric(5, 2), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "exercises",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("default_difficulty", _difficulty_level(), nullable=False),
    )

    op.create_table(
        "sessions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("exercise_id", sa.String(), sa.ForeignKey("exercises.id"), nullable=False),
        sa.Column("difficulty", _difficulty_level(), nullable=False),
        sa.Column("status", _session_status(), nullable=False),
        sa.Column("target_sets", sa.Integer(), nullable=False),
        sa.Column("target_reps", sa.Integer(), nullable=False),
        sa.Column("planned_rest_seconds", sa.Integer(), nullable=False),
        sa.Column("external_load_kg", sa.Numeric(6, 2), nullable=True),
        sa.Column("body_weight_kg", sa.Numeric(6, 2), nullable=True),
        sa.Column("total_reps", sa.Integer(), nullable=False),
        sa.Column("avg_form_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "session_feedback",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("exercise_id", sa.String(), sa.ForeignKey("exercises.id"), nullable=False),
        sa.Column(
            "session_id",
            sa.Uuid(),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("overall_rating", sa.Integer(), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("errors_count", sa.Integer(), nullable=False),
        sa.Column("top_errors", sa.JSON(), nullable=False),
        sa.Column("error_breakdown", sa.JSON(), nullable=False),
        sa.Column("action_items", sa.JSON(), nullable=False),
        sa.Column("confidence_overall", sa.Numeric(4, 3), nullable=True),
        sa.Column("signals_used", sa.JSON(), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "readiness_features_daily",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("exercise_id", sa.String(), sa.ForeignKey("exercises.id"), nullable=False),
        sa.Column("feature_date", sa.Date(), nullable=False),
        sa.Column("sessions_7d", sa.Integer(), nullable=False),
        sa.Column("reps_7d", sa.Integer(), nullable=False),
        sa.Column("avg_form_score_7d", sa.Numeric(5, 2), nullable=True),
        sa.Column("avg_rom_7d", sa.Numeric(7, 3), nullable=True),
        sa.Column("rom_trend_7d", sa.Numeric(7, 3), nullable=True),
        sa.Column("avg_rep_duration_ms_7d", sa.Numeric(10, 2), nullable=True),
        sa.Column("days_since_last_session", sa.Integer(), nullable=True),
        sa.Column("sleep_hours", sa.Numeric(4, 2), nullable=True),
        sa.Column("soreness_score", sa.Integer(), nullable=True),
        sa.Column("stress_score", sa.Integer(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "user_id",
            "exercise_id",
            "feature_date",
            name="uq_readiness_features_daily_user_exercise_date",
        ),
    )

    op.create_table(
        "rep_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            sa.Uuid(),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rep_number", sa.Integer(), nullable=False),
        sa.Column("min_angle", sa.Numeric(6, 2), nullable=True),
        sa.Column("max_angle", sa.Numeric(6, 2), nullable=True),
        sa.Column("range_of_motion", sa.Numeric(6, 2), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("depth_quality", sa.String(32), nullable=True),
        sa.Column("form_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("extra", sa.JSON(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "fatigue_predictions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("exercise_id", sa.String(), sa.ForeignKey("exercises.id"), nullable=False),
        sa.Column("readiness_score", sa.Integer(), nullable=False),
        sa.Column("fatigue_level", _fatigue_level(), nullable=False),
        sa.Column("recommendation", sa.Text(), nullable=False),
        sa.Column("factors", sa.JSON(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("input_snapshot", sa.JSON(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "fatigue_calibration_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("exercise_id", sa.String(), sa.ForeignKey("exercises.id"), nullable=False),
        sa.Column(
            "session_id",
            sa.Uuid(),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "prediction_id",
            sa.Uuid(),
            sa.ForeignKey("fatigue_predictions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("predicted_readiness_score", sa.Integer(), nullable=True),
        sa.Column("predicted_fatigue_level", sa.String(16), nullable=True),
        sa.Column("actual_performance_score", sa.Numeric(5, 2), nullable=False),
        sa.Column("readiness_delta", sa.Integer(), nullable=False),
        sa.Column("performance_drop_observed", sa.Boolean(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "achievements",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("xp_reward", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "user_achievements",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "achievement_id",
            sa.String(),
            sa.ForeignKey("achievements.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("unlocked_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "streaks",
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("current_streak_days", sa.Integer(), nullable=False),
        sa.Column("longest_streak_days", sa.Integer(), nullable=False),
        sa.Column("last_activity_date", sa.Date(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "xp_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            sa.Uuid(),
            sa.ForeignKey("sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("points", sa.Integer(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    bind = op.get_bind()
    for table in (
        "xp_events",
        "streaks",
        "user_achievements",
        "achievements",
        "fatigue_calibration_events",
        "fatigue_predictions",
        "rep_events",
        "readiness_features_daily",
        "session_feedback",
        "sessions",
        "exercises",
        "body_metrics",
        "users",
    ):
        op.drop_table(table)
    online = not context.is_offline_mode()
    _fatigue_level().drop(bind, checkfirst=online)
    _session_status().drop(bind, checkfirst=online)
    _difficulty_level().drop(bind, checkfirst=online)
