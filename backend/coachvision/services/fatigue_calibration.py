"""Post-session calibration: predicted readiness vs realized session quality."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from ..db.models import (
    FatigueCalibrationEvent,
    FatiguePrediction,
    RepEvent,
    Session as WorkoutSession,
)


def _reference_predicted_score() -> int:
    """When no pre-workout prediction exists, compare against a neutral baseline."""
    return 70


def compute_actual_performance_score(
    session: WorkoutSession,
    rep_rows: list[RepEvent],
) -> tuple[float, dict[str, Any]]:
    """
    Composite 0–100: form/depth quality, volume vs target, within-session ROM fade.
    """
    detail: dict[str, Any] = {}
    parts: list[float] = []

    if session.avg_form_score is not None:
        f = float(session.avg_form_score)
        parts.append(f)
        detail["avg_form_score_session"] = f

    good = shallow = very = 0
    roms: list[float] = []
    for r in rep_rows:
        if r.depth_quality == "good":
            good += 1
        elif r.depth_quality == "shallow":
            shallow += 1
        elif r.depth_quality == "very_shallow":
            very += 1
        if r.range_of_motion is not None:
            roms.append(float(r.range_of_motion))

    depth_total = good + shallow + very
    if depth_total > 0:
        depth_score = 100.0 * good / depth_total + 40.0 * shallow / depth_total
        parts.append(depth_score)
        detail["depth_quality_breakdown"] = {"good": good, "shallow": shallow, "very_shallow": very}

    target = max(1, (session.target_sets or 1) * (session.target_reps or 1))
    tr = session.total_reps or 0
    vol_score = min(100.0, 100.0 * tr / target)
    detail["volume_vs_target"] = {"total_reps": tr, "target_reps": target, "vol_score": round(vol_score, 2)}

    if parts:
        base = sum(parts) / len(parts)
    else:
        base = vol_score
    score = 0.55 * base + 0.45 * vol_score

    if len(roms) >= 6:
        first = sum(roms[:3]) / 3
        last = sum(roms[-3:]) / 3
        if first > 1e-6 and last < first * 0.88:
            score -= 12.0
            detail["intra_session_rom_fade"] = True
            detail["rom_first3_avg"] = round(first, 2)
            detail["rom_last3_avg"] = round(last, 2)

    score = max(0.0, min(100.0, score))
    detail["composite"] = round(score, 2)
    return round(score, 2), detail


def find_pre_session_prediction(
    db: DbSession,
    user_id: UUID,
    exercise_id: str,
    session_started_at: datetime | None,
    session_created_at: datetime | None,
) -> FatiguePrediction | None:
    """Latest fatigue prediction at or before workout start, within a 48h lookback."""
    anchor = session_started_at or session_created_at or datetime.now(timezone.utc)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    lo = anchor - timedelta(hours=48)
    return db.scalars(
        select(FatiguePrediction)
        .where(
            FatiguePrediction.user_id == user_id,
            FatiguePrediction.exercise_id == exercise_id,
            FatiguePrediction.generated_at <= anchor,
            FatiguePrediction.generated_at >= lo,
        )
        .order_by(FatiguePrediction.generated_at.desc())
        .limit(1)
    ).first()


def record_calibration_for_session(db: DbSession, workout: WorkoutSession) -> FatigueCalibrationEvent | None:
    if workout.status != "completed":
        return None

    if db.scalars(
        select(FatigueCalibrationEvent).where(FatigueCalibrationEvent.session_id == workout.id).limit(1)
    ).first():
        return None

    reps = list(
        db.scalars(select(RepEvent).where(RepEvent.session_id == workout.id)).all()
    )
    actual, perf_detail = compute_actual_performance_score(workout, reps)

    pred = find_pre_session_prediction(
        db,
        workout.user_id,
        workout.exercise_id,
        workout.started_at,
        workout.created_at,
    )

    predicted_score = pred.readiness_score if pred else None
    pred_level = pred.fatigue_level if pred else None
    ref = predicted_score if predicted_score is not None else _reference_predicted_score()
    delta = int(round(actual - ref))

    # Label a "drop" when the session was objectively poor, or when it
    # underperformed the (or a neutral) expectation by a clear margin.
    drop = actual < 45 or actual < ref - 12

    detail: dict[str, Any] = {
        "performance_breakdown": perf_detail,
        "had_pre_session_prediction": pred is not None,
        "prediction_generated_at": pred.generated_at.isoformat() if pred else None,
        "interpretation": _interpretation(predicted_score, pred_level, actual, delta),
    }

    ev = FatigueCalibrationEvent(
        user_id=workout.user_id,
        exercise_id=workout.exercise_id,
        session_id=workout.id,
        prediction_id=pred.id if pred else None,
        predicted_readiness_score=predicted_score,
        predicted_fatigue_level=pred_level,
        actual_performance_score=actual,
        readiness_delta=delta,
        performance_drop_observed=drop,
        detail=detail,
    )
    db.add(ev)
    return ev


def _interpretation(
    predicted: int | None,
    pred_level: str | None,
    actual: float,
    delta: int,
) -> str:
    if predicted is None:
        return (
            f"No pre-workout prediction; actual session quality score {actual:.0f}/100 "
            f"(vs neutral baseline {_reference_predicted_score()}), delta {delta:+d}."
        )
    if delta >= 10:
        return (
            f"Session quality ({actual:.0f}) exceeded predicted readiness ({predicted}); "
            "you may tolerate more load than the rule model expected."
        )
    if delta <= -12:
        return (
            f"Session quality ({actual:.0f}) underperformed predicted readiness ({predicted}); "
            "consider more conservative targets next time."
        )
    return (
        f"Session quality ({actual:.0f}) was close to predicted readiness ({predicted}, {pred_level})."
    )
