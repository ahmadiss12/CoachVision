"""Upsert readiness_features_daily after completed sessions."""

from __future__ import annotations

import datetime as dt
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from ..db.models import ReadinessFeaturesDaily
from .fatigue_features import fetch_rolling_features, last_self_report_from_predictions, snapshot_to_dict


def _parse_self_report_numeric(ctx: dict[str, Any]) -> tuple[float | None, int | None, int | None]:
    sleep = ctx.get("sleepHours", ctx.get("sleep_hours"))
    sore = ctx.get("muscleSoreness", ctx.get("soreness", ctx.get("muscle_soreness")))
    stress = ctx.get("stress")
    sleep_f: float | None = None
    if sleep is not None:
        try:
            sleep_f = float(sleep)
        except (TypeError, ValueError):
            pass
    sore_i: int | None = None
    if sore is not None:
        try:
            sore_i = max(1, min(5, int(sore)))
        except (TypeError, ValueError):
            pass
    stress_i: int | None = None
    if stress is not None:
        try:
            stress_i = max(1, min(5, int(stress)))
        except (TypeError, ValueError):
            pass
    return sleep_f, sore_i, stress_i


def upsert_readiness_features_daily(
    db: DbSession,
    user_id: UUID,
    exercise_id: str,
    *,
    feature_date: dt.date,
    as_of: datetime,
    current_session_id: UUID | None,
    window_days: int = 7,
) -> ReadinessFeaturesDaily:
    snap = fetch_rolling_features(
        db,
        user_id,
        exercise_id,
        window_days=window_days,
        as_of=as_of,
        current_session_id=current_session_id,
    )
    ctx = last_self_report_from_predictions(db, user_id, exercise_id, before=as_of)
    sleep_f, sore_i, stress_i = _parse_self_report_numeric(ctx)

    row = db.scalars(
        select(ReadinessFeaturesDaily).where(
            ReadinessFeaturesDaily.user_id == user_id,
            ReadinessFeaturesDaily.exercise_id == exercise_id,
            ReadinessFeaturesDaily.feature_date == feature_date,
        )
    ).first()

    now = datetime.now(timezone.utc)
    payload = {
        "sessions_7d": snap.sessions_7d,
        "reps_7d": snap.reps_7d,
        "avg_form_score_7d": snap.avg_form_score_7d,
        "avg_rom_7d": snap.avg_rom_7d,
        "rom_trend_7d": snap.rom_trend_7d,
        "avg_rep_duration_ms_7d": snap.avg_rep_duration_ms_7d,
        "days_since_last_session": snap.days_since_last_session,
        "sleep_hours": sleep_f,
        "soreness_score": sore_i,
        "stress_score": stress_i,
        "updated_at": now,
    }

    if row is None:
        row = ReadinessFeaturesDaily(
            user_id=user_id,
            exercise_id=exercise_id,
            feature_date=feature_date,
            **payload,
        )
        db.add(row)
    else:
        for k, v in payload.items():
            setattr(row, k, v)
        db.add(row)
    db.flush()
    return row


def snapshot_for_api(db: DbSession, user_id: UUID, exercise_id: str, window_days: int) -> dict[str, Any]:
    """Latest daily row if any, merged with live rolling snapshot for transparency."""
    today = datetime.now(timezone.utc).date()
    daily = db.scalars(
        select(ReadinessFeaturesDaily)
        .where(
            ReadinessFeaturesDaily.user_id == user_id,
            ReadinessFeaturesDaily.exercise_id == exercise_id,
        )
        .order_by(ReadinessFeaturesDaily.feature_date.desc())
        .limit(1)
    ).first()
    live = fetch_rolling_features(db, user_id, exercise_id, window_days=window_days)
    out = snapshot_to_dict(live)
    out["window_days"] = window_days
    if daily:
        out["last_daily_rollup_date"] = str(daily.feature_date)
        out["stored_daily"] = {
            "sessions_7d": daily.sessions_7d,
            "reps_7d": daily.reps_7d,
            "avg_form_score_7d": float(daily.avg_form_score_7d)
            if daily.avg_form_score_7d is not None
            else None,
        }
    return out
