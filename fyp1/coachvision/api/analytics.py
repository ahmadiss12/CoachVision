"""Analytics endpoints for dashboard/history."""

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from .deps import get_current_user
from .schemas import AnalyticsOverviewResponse
from ..db.models import Session as WorkoutSession, User
from ..db.session import get_db

router = APIRouter(prefix="/analytics")


@router.get("/overview", response_model=AnalyticsOverviewResponse)
def analytics_overview(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalyticsOverviewResponse:
    total_sessions = db.query(func.count(WorkoutSession.id)).filter(
        WorkoutSession.user_id == current_user.id
    ).scalar() or 0

    completed_sessions = db.query(func.count(WorkoutSession.id)).filter(
        WorkoutSession.user_id == current_user.id,
        WorkoutSession.status == "completed",
    ).scalar() or 0

    avg_form_score = db.query(func.avg(WorkoutSession.avg_form_score)).filter(
        WorkoutSession.user_id == current_user.id,
        WorkoutSession.avg_form_score.is_not(None),
    ).scalar()

    return AnalyticsOverviewResponse(
        totalSessions=int(total_sessions),
        completedSessions=int(completed_sessions),
        avgFormScore=float(avg_form_score) if avg_form_score is not None else None,
    )


@router.get("/exercise/{exercise_id}", response_model=AnalyticsOverviewResponse)
def analytics_exercise(
    exercise_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalyticsOverviewResponse:
    total_sessions = db.query(func.count(WorkoutSession.id)).filter(
        WorkoutSession.user_id == current_user.id,
        WorkoutSession.exercise_id == exercise_id,
    ).scalar() or 0

    completed_sessions = db.query(func.count(WorkoutSession.id)).filter(
        WorkoutSession.user_id == current_user.id,
        WorkoutSession.exercise_id == exercise_id,
        WorkoutSession.status == "completed",
    ).scalar() or 0

    avg_form_score = db.query(func.avg(WorkoutSession.avg_form_score)).filter(
        WorkoutSession.user_id == current_user.id,
        WorkoutSession.exercise_id == exercise_id,
        WorkoutSession.avg_form_score.is_not(None),
    ).scalar()

    return AnalyticsOverviewResponse(
        totalSessions=int(total_sessions),
        completedSessions=int(completed_sessions),
        avgFormScore=float(avg_form_score) if avg_form_score is not None else None,
    )

