"""Run fatigue aggregation + calibration after a session is marked completed."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session as DbSession

from ..db.models import Session as WorkoutSession
from .fatigue_aggregation import upsert_readiness_features_daily
from .fatigue_calibration import record_calibration_for_session
from .gamification_service import GamificationService

logger = logging.getLogger(__name__)


def run_after_completed_session(db: DbSession, workout_id: UUID) -> None:
    """Safe to call after the workout + rep_events transaction has committed."""
    w = db.get(WorkoutSession, workout_id)
    if w is None or w.status != "completed":
        return
    try:
        as_of = w.ended_at or datetime.now(timezone.utc)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        upsert_readiness_features_daily(
            db,
            w.user_id,
            w.exercise_id,
            feature_date=as_of.date(),
            as_of=as_of,
            current_session_id=w.id,
        )
        record_calibration_for_session(db, w)
        GamificationService(db).apply_on_completed_session(w.id)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("fatigue post-session pipeline failed for workout %s", workout_id)
