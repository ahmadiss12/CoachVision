"""Persist live WebSocket workout state to the database."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import delete
from sqlalchemy.orm import Session as DbSession

from ..db.models import Exercise, RepEvent, Session as WorkoutSession
from ..db.models import SessionFeedback
from .session_feedback_generator import build_session_feedback_row


def normalize_exercise_id(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


_VALID_DEPTH = frozenset({"good", "shallow", "very_shallow"})
_REP_CORE_KEYS = frozenset(
    {
        "rep_number",
        "min_angle",
        "max_angle",
        "min_elbow_angle",
        "max_elbow_angle",
        "range_of_motion",
        "duration",
        "depth_quality",
    }
)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
        if x != x or x in (float("inf"), float("-inf")):  # noqa: PLR0124
            return None
        return x
    except (TypeError, ValueError):
        return None


def _insert_rep_events_for_session(
    db: DbSession,
    session_id: UUID,
    rep_metrics: list[Any],
) -> None:
    """Map counter rep_metrics (varies by exercise) into rep_events rows."""
    for i, rep in enumerate(rep_metrics):
        if not isinstance(rep, dict):
            continue
        rep_number = int(rep.get("rep_number", i + 1))
        min_a = _optional_float(
            rep.get("min_angle", rep.get("min_elbow_angle"))
        )
        max_a = _optional_float(
            rep.get("max_angle", rep.get("max_elbow_angle"))
        )
        rom = _optional_float(rep.get("range_of_motion"))
        duration = rep.get("duration")
        duration_ms: int | None = None
        if duration is not None:
            try:
                duration_ms = max(0, int(float(duration) * 1000))
            except (TypeError, ValueError):
                duration_ms = None
        dq_raw = rep.get("depth_quality")
        depth_quality: str | None = None
        if isinstance(dq_raw, str) and dq_raw in _VALID_DEPTH:
            depth_quality = dq_raw
        extra = {k: v for k, v in rep.items() if k not in _REP_CORE_KEYS}
        db.add(
            RepEvent(
                session_id=session_id,
                rep_number=rep_number,
                min_angle=min_a,
                max_angle=max_a,
                range_of_motion=rom,
                duration_ms=duration_ms,
                depth_quality=depth_quality,
                form_score=_optional_float(rep.get("form_score")),
                extra=extra,
            )
        )


def finalize_completed_workout(
    db: DbSession,
    workout: WorkoutSession,
    export_data: dict[str, Any],
) -> None:
    """Apply dispatcher export_session_data() to the sessions row."""
    now = datetime.now(timezone.utc)
    inner = export_data.get("summary") or {}
    total_reps = int(inner.get("total_reps", 0))
    rep_metrics = export_data.get("rep_metrics") or []
    if total_reps <= 0 and rep_metrics:
        total_reps = len(rep_metrics)

    workout.status = "completed"
    workout.ended_at = now
    workout.total_reps = total_reps
    if workout.started_at:
        workout.duration_seconds = max(0, int((now - workout.started_at).total_seconds()))

    good = inner.get("good_reps")
    if total_reps > 0 and good is not None:
        workout.avg_form_score = round(100.0 * int(good) / total_reps, 2)

    workout.updated_at = now
    db.add(workout)
    if rep_metrics:
        _insert_rep_events_for_session(db, workout.id, rep_metrics)

    db.execute(delete(SessionFeedback).where(SessionFeedback.session_id == workout.id))
    db.add(build_session_feedback_row(workout, export_data, now=now))

    db.commit()

    from .fatigue_post_session import run_after_completed_session

    run_after_completed_session(db, workout.id)


def abort_active_workout(db: DbSession, workout: WorkoutSession | None) -> None:
    """Mark active/planned WS-linked session as aborted (client dropped)."""
    if workout is None:
        return
    row = db.get(WorkoutSession, workout.id)
    if row is None or row.status not in ("active", "planned"):
        return
    workout = row
    now = datetime.now(timezone.utc)
    workout.status = "aborted"
    workout.ended_at = now
    if workout.started_at:
        workout.duration_seconds = max(0, int((now - workout.started_at).total_seconds()))
    workout.updated_at = now
    db.add(workout)
    db.commit()


def load_or_create_workout_for_ws_start(
    db: DbSession,
    *,
    user_id: UUID,
    exercise_name: str,
    difficulty: str,
    rest_session_id: UUID | None,
    target_sets: int,
    target_reps: int,
    external_load_kg: float | None = None,
    body_weight_kg: float | None = None,
) -> tuple[WorkoutSession | None, str | None]:
    """
    Returns (workout, error_code). error_code is set on failure.
    """
    exercise_id = normalize_exercise_id(exercise_name)
    now = datetime.now(timezone.utc)

    if db.get(Exercise, exercise_id) is None:
        return None, "EXERCISE_NOT_FOUND"

    if rest_session_id is not None:
        row = db.get(WorkoutSession, rest_session_id)
        if row is None or row.user_id != user_id:
            return None, "SESSION_NOT_FOUND"
        if row.status == "completed":
            return None, "SESSION_ALREADY_COMPLETED"
        if row.exercise_id != exercise_id:
            return None, "EXERCISE_MISMATCH"
        row.status = "active"
        if not row.started_at:
            row.started_at = now
        if external_load_kg is not None:
            row.external_load_kg = external_load_kg
        if body_weight_kg is not None:
            row.body_weight_kg = body_weight_kg
        row.updated_at = now
        db.add(row)
        db.commit()
        db.refresh(row)
        return row, None

    workout = WorkoutSession(
        user_id=user_id,
        exercise_id=exercise_id,
        difficulty=difficulty,
        status="active",
        target_sets=max(1, target_sets),
        target_reps=max(1, target_reps),
        external_load_kg=external_load_kg,
        body_weight_kg=body_weight_kg,
        started_at=now,
        updated_at=now,
    )
    db.add(workout)
    db.commit()
    db.refresh(workout)
    return workout, None
