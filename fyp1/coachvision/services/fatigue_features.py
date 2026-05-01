"""Rolling feature computation for fatigue (sessions + rep_events)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from ..db.models import FatiguePrediction, RepEvent, Session as WorkoutSession


@dataclass
class RollingFeatureSnapshot:
    """Serializable bundle used by the rule engine and daily rollup."""

    sessions_7d: int
    reps_7d: int
    avg_form_score_7d: float | None
    avg_rom_7d: float | None
    rom_trend_7d: float | None
    avg_rep_duration_ms_7d: float | None
    days_since_last_session: int | None
    form_degradation_7d: float | None  # last2 session avg form - first2 avg form
    shallow_rep_ratio_7d: float | None  # proxy for "failure / quality drop"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fetch_rolling_features(
    db: DbSession,
    user_id: UUID,
    exercise_id: str,
    *,
    window_days: int = 7,
    as_of: datetime | None = None,
    current_session_id: UUID | None = None,
) -> RollingFeatureSnapshot:
    """
    Aggregate completed sessions in (as_of - window_days, as_of].
    If current_session_id is set, days_since_last_session excludes that session when
    finding the prior session (caller passes the session just completed for 'prior' gap).
    """
    as_of = as_of or _utc_now()
    window_start = as_of - timedelta(days=window_days)

    q = (
        select(WorkoutSession)
        .where(
            WorkoutSession.user_id == user_id,
            WorkoutSession.exercise_id == exercise_id,
            WorkoutSession.status == "completed",
            WorkoutSession.ended_at.is_not(None),
            WorkoutSession.ended_at >= window_start,
            WorkoutSession.ended_at <= as_of,
        )
        .order_by(WorkoutSession.ended_at.asc())
    )
    sessions = list(db.scalars(q).all())

    sessions_7d = len(sessions)
    reps_7d = int(sum(s.total_reps or 0 for s in sessions))

    form_scores = [float(s.avg_form_score) for s in sessions if s.avg_form_score is not None]
    avg_form = sum(form_scores) / len(form_scores) if form_scores else None

    session_ids = [s.id for s in sessions]
    roms_by_session: dict[UUID, list[float]] = {}
    durs_ms: list[int] = []
    shallow = 0
    total_depth = 0

    if session_ids:
        rep_rows = db.scalars(
            select(RepEvent).where(RepEvent.session_id.in_(session_ids))
        ).all()
        for r in rep_rows:
            if r.range_of_motion is not None:
                roms_by_session.setdefault(r.session_id, []).append(float(r.range_of_motion))
            if r.duration_ms is not None:
                durs_ms.append(int(r.duration_ms))
            if r.depth_quality:
                total_depth += 1
                if r.depth_quality in ("shallow", "very_shallow"):
                    shallow += 1

    flat_roms: list[float] = []
    for sid in session_ids:
        for v in roms_by_session.get(sid, []):
            flat_roms.append(v)

    avg_rom = sum(flat_roms) / len(flat_roms) if flat_roms else None

    # Per-session mean ROM (chronological), then trend = mean(last two) - mean(first two)
    rom_trend: float | None = None
    per_sess_rom: list[float] = []
    for s in sessions:
        vals = roms_by_session.get(s.id, [])
        if vals:
            per_sess_rom.append(sum(vals) / len(vals))
    if len(per_sess_rom) >= 4:
        first = sum(per_sess_rom[:2]) / 2
        last = sum(per_sess_rom[-2:]) / 2
        rom_trend = round(last - first, 3)
    elif len(per_sess_rom) >= 2:
        rom_trend = round(per_sess_rom[-1] - per_sess_rom[0], 3)

    avg_dur_ms = sum(durs_ms) / len(durs_ms) if durs_ms else None

    shallow_ratio = (shallow / total_depth) if total_depth else None

    # Form degradation: compare avg form of last 2 vs first 2 sessions in window
    form_deg: float | None = None
    if len(form_scores) >= 4:
        form_deg = round(
            (sum(form_scores[-2:]) / 2) - (sum(form_scores[:2]) / 2),
            2,
        )
    elif len(form_scores) >= 2:
        form_deg = round(form_scores[-1] - form_scores[0], 2)

    # Days since previous completed session (strictly before this workout when current_session_id set)
    days_since: int | None = None
    prior_stmt = select(WorkoutSession).where(
        WorkoutSession.user_id == user_id,
        WorkoutSession.exercise_id == exercise_id,
        WorkoutSession.status == "completed",
        WorkoutSession.ended_at.is_not(None),
        WorkoutSession.ended_at <= as_of,
    )
    if current_session_id is not None:
        prior_stmt = prior_stmt.where(WorkoutSession.id != current_session_id)
    prior_stmt = prior_stmt.order_by(WorkoutSession.ended_at.desc()).limit(1)
    prior_row = db.scalars(prior_stmt).first()
    if prior_row and prior_row.ended_at is not None:
        days_since = max(0, (as_of.date() - prior_row.ended_at.date()).days)

    return RollingFeatureSnapshot(
        sessions_7d=sessions_7d,
        reps_7d=reps_7d,
        avg_form_score_7d=round(avg_form, 2) if avg_form is not None else None,
        avg_rom_7d=round(avg_rom, 3) if avg_rom is not None else None,
        rom_trend_7d=rom_trend,
        avg_rep_duration_ms_7d=round(avg_dur_ms, 2) if avg_dur_ms is not None else None,
        days_since_last_session=days_since,
        form_degradation_7d=form_deg,
        shallow_rep_ratio_7d=round(shallow_ratio, 3) if shallow_ratio is not None else None,
    )


def snapshot_to_dict(s: RollingFeatureSnapshot) -> dict[str, Any]:
    return {
        "sessions_7d": s.sessions_7d,
        "reps_7d": s.reps_7d,
        "avg_form_score_7d": s.avg_form_score_7d,
        "avg_rom_7d": s.avg_rom_7d,
        "rom_trend_7d": s.rom_trend_7d,
        "avg_rep_duration_ms_7d": s.avg_rep_duration_ms_7d,
        "days_since_last_session": s.days_since_last_session,
        "form_degradation_7d": s.form_degradation_7d,
        "shallow_rep_ratio_7d": s.shallow_rep_ratio_7d,
    }


def last_self_report_from_predictions(
    db: DbSession,
    user_id: UUID,
    exercise_id: str,
    before: datetime | None = None,
) -> dict[str, Any]:
    """Latest userContext from fatigue_predictions.input_snapshot (before optional instant)."""
    before = before or _utc_now()
    row = db.scalars(
        select(FatiguePrediction)
        .where(
            FatiguePrediction.user_id == user_id,
            FatiguePrediction.exercise_id == exercise_id,
            FatiguePrediction.generated_at <= before,
        )
        .order_by(FatiguePrediction.generated_at.desc())
        .limit(1)
    ).first()
    if not row or not row.input_snapshot:
        return {}
    ctx = row.input_snapshot.get("userContext") or row.input_snapshot.get("user_context") or {}
    return ctx if isinstance(ctx, dict) else {}
