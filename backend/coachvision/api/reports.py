"""Coach-facing report endpoints."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from .deps import get_current_user
from .schemas import (
    DailyReportResponse,
    DailyReportSessionResponse,
    DailyReportSummaryResponse,
    ReportRepStatsResponse,
    ReportUserResponse,
    SessionFeedbackResponse,
)
from ..db.models import Exercise, RepEvent, Session, SessionFeedback, User
from ..db.session import get_db

router = APIRouter(prefix="/reports")

_HOLD_EXERCISE_IDS = frozenset({"plank", "wall_sit"})


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):  # noqa: PLR0124
        return None
    return parsed


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _tz_for_user(user: User) -> ZoneInfo:
    try:
        return ZoneInfo(user.timezone or "UTC")
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def _day_bounds_utc(report_date: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start_local = datetime.combine(report_date, time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _exercise_name(exercise_id: str, exercise_by_id: dict[str, Exercise]) -> str:
    exercise = exercise_by_id.get(exercise_id)
    if exercise is not None:
        return exercise.name
    return " ".join(part.capitalize() for part in exercise_id.replace("-", "_").split("_") if part)


def _rep_stats(rows: list[RepEvent]) -> ReportRepStatsResponse:
    rom_values = [_to_float(row.range_of_motion) for row in rows]
    duration_values = [_to_float(row.duration_ms) for row in rows]
    score_values = [_to_float(row.form_score) for row in rows]
    depth_counts = Counter(row.depth_quality for row in rows if row.depth_quality)

    return ReportRepStatsResponse(
        rep_count=len(rows),
        avg_range_of_motion=_avg([value for value in rom_values if value is not None]),
        avg_rep_duration_ms=_avg([value for value in duration_values if value is not None]),
        avg_form_score=_avg([value for value in score_values if value is not None]),
        depth_quality_counts=dict(depth_counts),
    )


@router.get("/daily", response_model=DailyReportResponse)
def get_daily_report(
    report_date: date | None = Query(default=None, alias="date"),
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DailyReportResponse:
    tz = _tz_for_user(current_user)
    selected_date = report_date or datetime.now(tz).date()
    start_utc, end_utc = _day_bounds_utc(selected_date, tz)
    generated_at = datetime.now(timezone.utc)

    sessions = db.scalars(
        select(Session)
        .where(
            Session.user_id == current_user.id,
            Session.status == "completed",
            Session.ended_at.is_not(None),
            Session.ended_at >= start_utc,
            Session.ended_at < end_utc,
        )
        .order_by(Session.ended_at.asc(), Session.created_at.asc())
    ).all()

    session_ids: list[UUID] = [row.id for row in sessions]
    exercise_ids = {row.exercise_id for row in sessions}

    exercises = []
    if exercise_ids:
        exercises = db.scalars(select(Exercise).where(Exercise.id.in_(exercise_ids))).all()
    exercise_by_id = {row.id: row for row in exercises}

    feedback_by_session: dict[UUID, SessionFeedback] = {}
    rep_events_by_session: dict[UUID, list[RepEvent]] = defaultdict(list)
    if session_ids:
        feedback_rows = db.scalars(
            select(SessionFeedback).where(SessionFeedback.session_id.in_(session_ids))
        ).all()
        feedback_by_session = {row.session_id: row for row in feedback_rows}

        rep_rows = db.scalars(select(RepEvent).where(RepEvent.session_id.in_(session_ids))).all()
        for row in rep_rows:
            rep_events_by_session[row.session_id].append(row)

    report_sessions: list[DailyReportSessionResponse] = []
    score_values: list[float] = []
    total_reps = 0
    total_duration_seconds = 0
    volume_loads: list[float] = []

    for row in sessions:
        total_reps += row.total_reps or 0
        total_duration_seconds += row.duration_seconds or 0
        score = _to_float(row.avg_form_score)
        if score is not None:
            score_values.append(score)

        external_load = _to_float(row.external_load_kg)
        body_weight = _to_float(row.body_weight_kg)
        uses_external_load = bool(external_load and external_load > 0)
        estimated_volume_load = None
        if uses_external_load and row.exercise_id not in _HOLD_EXERCISE_IDS and row.total_reps:
            estimated_volume_load = round(external_load * row.total_reps, 2)
            volume_loads.append(estimated_volume_load)

        feedback_row = feedback_by_session.get(row.id)
        report_sessions.append(
            DailyReportSessionResponse(
                session_id=row.id,
                exercise_id=row.exercise_id,
                exercise_name=_exercise_name(row.exercise_id, exercise_by_id),
                difficulty=row.difficulty,
                target_sets=row.target_sets,
                target_reps=row.target_reps,
                total_reps=row.total_reps,
                avg_form_score=score,
                duration_seconds=row.duration_seconds,
                started_at=row.started_at,
                ended_at=row.ended_at,
                external_load_kg=external_load,
                body_weight_kg=body_weight,
                uses_external_load=uses_external_load,
                estimated_volume_load_kg=estimated_volume_load,
                feedback=SessionFeedbackResponse.model_validate(feedback_row)
                if feedback_row is not None
                else None,
                rep_stats=_rep_stats(rep_events_by_session.get(row.id, [])),
            )
        )

    return DailyReportResponse(
        generated_at=generated_at,
        user=ReportUserResponse(
            id=current_user.id,
            email=current_user.email,
            display_name=current_user.display_name,
        ),
        summary=DailyReportSummaryResponse(
            date=selected_date,
            timezone=str(tz),
            total_sessions=len(report_sessions),
            total_reps=total_reps,
            total_duration_seconds=total_duration_seconds,
            avg_form_score=_avg(score_values),
            total_external_load_volume_kg=round(sum(volume_loads), 2) if volume_loads else None,
        ),
        sessions=report_sessions,
    )
