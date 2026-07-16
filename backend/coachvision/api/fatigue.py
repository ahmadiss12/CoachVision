"""Pre-exercise fatigue/readiness endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from .deps import get_current_user
from .schemas import (
    ExplainabilityFactorResponse,
    FatiguePredictRequest,
    FatiguePredictResponse,
)
from ..db.models import FatiguePrediction, User
from ..db.session import get_db
from ..services.fatigue_features import fetch_rolling_features
from ..services.readiness_model import predict_readiness

router = APIRouter(prefix="/fatigue")


def _explainability_from_snapshot(snap: dict | None) -> list[ExplainabilityFactorResponse]:
    if not snap:
        return []
    raw = snap.get("explainability") or []
    out: list[ExplainabilityFactorResponse] = []
    for item in raw:
        if isinstance(item, dict) and "key" in item:
            try:
                out.append(ExplainabilityFactorResponse.model_validate(item))
            except Exception:  # noqa: BLE001
                continue
    return out


def _feature_snapshot_from_row(row: FatiguePrediction) -> dict:
    snap = row.input_snapshot or {}
    return snap.get("feature_snapshot") or {}


@router.post("/predict", response_model=FatiguePredictResponse)
def predict_fatigue(
    payload: FatiguePredictRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FatiguePredictResponse:
    rolling = fetch_rolling_features(
        db,
        current_user.id,
        payload.exercise_id,
        window_days=payload.recent_window_days,
    )
    result, model_version = predict_readiness(
        rolling,
        payload.user_context,
        window_days=payload.recent_window_days,
        exercise_id=payload.exercise_id,
    )

    record = FatiguePrediction(
        user_id=current_user.id,
        exercise_id=payload.exercise_id,
        readiness_score=result.readiness_score,
        fatigue_level=result.fatigue_level,
        recommendation=result.recommendation,
        factors=result.factors,
        model_version=model_version,
        input_snapshot={
            "recentWindowDays": payload.recent_window_days,
            "userContext": payload.user_context,
            "explainability": [e.to_dict() for e in result.explainability],
            "feature_snapshot": result.feature_snapshot,
        },
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return FatiguePredictResponse(
        exerciseId=record.exercise_id,
        readinessScore=record.readiness_score,
        fatigueLevel=record.fatigue_level,
        recommendation=record.recommendation,
        factors=record.factors,
        generatedAt=record.generated_at or datetime.now(timezone.utc),
        predictionId=record.id,
        explainability=[
            ExplainabilityFactorResponse.model_validate(e.to_dict())
            for e in result.explainability
        ],
        featureSnapshot=result.feature_snapshot,
    )


@router.get("/history", response_model=list[FatiguePredictResponse])
def fatigue_history(
    exerciseId: str,  # noqa: N803
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[FatiguePredictResponse]:
    rows = db.scalars(
        select(FatiguePrediction)
        .where(
            FatiguePrediction.user_id == current_user.id,
            FatiguePrediction.exercise_id == exerciseId,
        )
        .order_by(FatiguePrediction.generated_at.desc())
        .limit(max(1, min(limit, 100)))
    ).all()

    out: list[FatiguePredictResponse] = []
    for row in rows:
        snap = row.input_snapshot or {}
        out.append(
            FatiguePredictResponse(
                exerciseId=row.exercise_id,
                readinessScore=row.readiness_score,
                fatigueLevel=row.fatigue_level,
                recommendation=row.recommendation,
                factors=row.factors,
                generatedAt=row.generated_at,
                predictionId=row.id,
                explainability=_explainability_from_snapshot(snap),
                featureSnapshot=_feature_snapshot_from_row(row),
            )
        )
    return out
