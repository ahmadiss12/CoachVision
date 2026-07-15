"""SQLAlchemy models for Option C API endpoints."""

from __future__ import annotations

import datetime as dt
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import JSON, BigInteger, Boolean, Date, DateTime, Enum, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def _empty_list() -> list:
    return []


def _empty_dict() -> dict:
    return {}


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(
        Enum("client", "trainer", "admin", name="user_role"),
        default="client",
        nullable=False,
    )
    avatar_url: Mapped[str | None] = mapped_column(String, nullable=True)
    date_of_birth: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    height_cm: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    weight_kg: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    body_fat_percent: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    timezone: Mapped[str] = mapped_column(String, default="UTC")
    locale: Mapped[str] = mapped_column(String, default="en")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    sessions: Mapped[list["Session"]] = relationship(back_populates="user")
    body_metrics: Mapped[list["BodyMetric"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class TrainerClient(Base):
    """Active coaching relationship between a trainer and a client."""

    __tablename__ = "trainer_clients"
    __table_args__ = (
        UniqueConstraint("trainer_id", "client_id", name="uq_trainer_clients_pair"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    trainer_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    client_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        Enum("active", "ended", name="trainer_client_status"),
        default="active",
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ClientInvite(Base):
    """Single-use invite code a trainer sends to onboard a client."""

    __tablename__ = "client_invites"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    trainer_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("pending", "accepted", "revoked", "expired", name="invite_status"),
        default="pending",
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    accepted_by: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Program(Base):
    """A reusable training program built by a trainer."""

    __tablename__ = "programs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    trainer_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    weeks: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    days: Mapped[list["ProgramDay"]] = relationship(
        back_populates="program",
        cascade="all, delete-orphan",
        order_by="ProgramDay.day_index",
    )


class ProgramDay(Base):
    """One day inside a program cycle (day_index is 1-based within the cycle)."""

    __tablename__ = "program_days"
    __table_args__ = (
        UniqueConstraint("program_id", "day_index", name="uq_program_days_program_day"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    program_id: Mapped[UUID] = mapped_column(
        ForeignKey("programs.id", ondelete="CASCADE"), nullable=False
    )
    day_index: Mapped[int] = mapped_column(Integer, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    program: Mapped["Program"] = relationship(back_populates="days")
    exercises: Mapped[list["ProgramExercise"]] = relationship(
        back_populates="day",
        cascade="all, delete-orphan",
        order_by="ProgramExercise.position",
    )


class ProgramExercise(Base):
    """A prescribed exercise inside a program day."""

    __tablename__ = "program_exercises"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    program_day_id: Mapped[UUID] = mapped_column(
        ForeignKey("program_days.id", ondelete="CASCADE"), nullable=False
    )
    exercise_id: Mapped[str] = mapped_column(ForeignKey("exercises.id"), nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    target_sets: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    target_reps: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    rest_seconds: Mapped[int] = mapped_column(Integer, default=60, nullable=False)
    external_load_kg: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    notes: Mapped[str | None] = mapped_column(String(300), nullable=True)

    day: Mapped["ProgramDay"] = relationship(back_populates="exercises")


class ProgramAssignment(Base):
    """A program assigned to a client, cycling from start_date."""

    __tablename__ = "program_assignments"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    program_id: Mapped[UUID] = mapped_column(
        ForeignKey("programs.id", ondelete="CASCADE"), nullable=False
    )
    trainer_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    client_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    start_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum("active", "completed", "cancelled", name="assignment_status"),
        default="active",
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    program: Mapped["Program"] = relationship()


class BodyMetric(Base):
    __tablename__ = "body_metrics"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    entry_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    body_fat_percent: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="body_metrics")


class Exercise(Base):
    __tablename__ = "exercises"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_difficulty: Mapped[str] = mapped_column(
        Enum("beginner", "intermediate", "advanced", name="difficulty_level"),
        default="intermediate",
        nullable=False,
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    exercise_id: Mapped[str] = mapped_column(ForeignKey("exercises.id"))
    assignment_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("program_assignments.id", ondelete="SET NULL"), nullable=True
    )
    difficulty: Mapped[str] = mapped_column(
        Enum("beginner", "intermediate", "advanced", name="difficulty_level"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        Enum("planned", "active", "completed", "aborted", name="session_status"),
        default="planned",
        nullable=False,
    )
    target_sets: Mapped[int] = mapped_column(Integer, default=1)
    target_reps: Mapped[int] = mapped_column(Integer, default=1)
    planned_rest_seconds: Mapped[int] = mapped_column(Integer, default=60)
    external_load_kg: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    body_weight_kg: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    total_reps: Mapped[int] = mapped_column(Integer, default=0)
    avg_form_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="sessions")
    session_feedback: Mapped["SessionFeedback | None"] = relationship(
        "SessionFeedback",
        back_populates="session",
        uselist=False,
    )


class SessionFeedback(Base):
    """One post-session human-readable feedback summary (v1: per session)."""

    __tablename__ = "session_feedback"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    exercise_id: Mapped[str] = mapped_column(ForeignKey("exercises.id"), nullable=False)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    overall_rating: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summary_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    errors_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    top_errors: Mapped[list] = mapped_column(JSON, default=_empty_list, nullable=False)
    error_breakdown: Mapped[dict] = mapped_column(JSON, default=_empty_dict, nullable=False)
    action_items: Mapped[list] = mapped_column(JSON, default=_empty_list, nullable=False)
    confidence_overall: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    signals_used: Mapped[dict] = mapped_column(JSON, default=_empty_dict, nullable=False)
    version: Mapped[str] = mapped_column(String(64), default="feedback_rule_v1", nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    session: Mapped["Session"] = relationship("Session", back_populates="session_feedback")


class ReadinessFeaturesDaily(Base):
    """Rolling aggregates persisted after sessions (one row per user/exercise/calendar day UTC)."""

    __tablename__ = "readiness_features_daily"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "exercise_id",
            "feature_date",
            name="uq_readiness_features_daily_user_exercise_date",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    exercise_id: Mapped[str] = mapped_column(ForeignKey("exercises.id"), nullable=False)
    feature_date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    sessions_7d: Mapped[int] = mapped_column(Integer, default=0)
    reps_7d: Mapped[int] = mapped_column(Integer, default=0)
    avg_form_score_7d: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    avg_rom_7d: Mapped[float | None] = mapped_column(Numeric(7, 3), nullable=True)
    rom_trend_7d: Mapped[float | None] = mapped_column(Numeric(7, 3), nullable=True)
    avg_rep_duration_ms_7d: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    days_since_last_session: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sleep_hours: Mapped[float | None] = mapped_column(Numeric(4, 2), nullable=True)
    soreness_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stress_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class RepEvent(Base):
    """Per-rep metrics persisted after a completed live session."""

    __tablename__ = "rep_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    rep_number: Mapped[int] = mapped_column(Integer, nullable=False)
    min_angle: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    max_angle: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    range_of_motion: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    depth_quality: Mapped[str | None] = mapped_column(String(32), nullable=True)
    form_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class FatiguePrediction(Base):
    __tablename__ = "fatigue_predictions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    exercise_id: Mapped[str] = mapped_column(ForeignKey("exercises.id"), nullable=False)
    readiness_score: Mapped[int] = mapped_column(Integer, nullable=False)
    fatigue_level: Mapped[str] = mapped_column(
        Enum("low", "moderate", "high", name="fatigue_level"),
        nullable=False,
    )
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    factors: Mapped[list[str]] = mapped_column(JSON, default=list)
    model_version: Mapped[str] = mapped_column(String, default="rule_v1")
    input_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class FatigueCalibrationEvent(Base):
    """Compares pre-workout prediction (if any) to realized session performance."""

    __tablename__ = "fatigue_calibration_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    exercise_id: Mapped[str] = mapped_column(ForeignKey("exercises.id"), nullable=False)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    prediction_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("fatigue_predictions.id", ondelete="SET NULL"),
        nullable=True,
    )
    predicted_readiness_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    predicted_fatigue_level: Mapped[str | None] = mapped_column(String(16), nullable=True)
    actual_performance_score: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    readiness_delta: Mapped[int] = mapped_column(Integer, nullable=False)
    performance_drop_observed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class Achievement(Base):
    __tablename__ = "achievements"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    xp_reward: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class UserAchievement(Base):
    __tablename__ = "user_achievements"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    achievement_id: Mapped[str] = mapped_column(
        ForeignKey("achievements.id", ondelete="CASCADE"), nullable=False
    )
    unlocked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class Streak(Base):
    __tablename__ = "streaks"

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    current_streak_days: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    longest_streak_days: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_activity_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class XpEvent(Base):
    __tablename__ = "xp_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    points: Mapped[int] = mapped_column(Integer, nullable=False)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

