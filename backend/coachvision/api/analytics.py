"""Analytics endpoints for dashboard/history."""

from fastapi import APIRouter, Depends
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from .deps import get_current_user
from .schemas import AnalyticsOverviewResponse
from ..db.models import Session as WorkoutSession, User
from ..db.session import get_db

router = APIRouter(prefix="/analytics")


def _overview(db: Session, user_id, exercise_id: str | None = None) -> AnalyticsOverviewResponse:
    query = select(
        func.count(WorkoutSession.id),
        func.count(case((WorkoutSession.status == "completed", 1))),
        func.avg(WorkoutSession.avg_form_score),
    ).where(WorkoutSession.user_id == user_id)
    if exercise_id is not None:
        query = query.where(WorkoutSession.exercise_id == exercise_id)

    total, completed, avg_form = db.execute(query).one()
    return AnalyticsOverviewResponse(
        totalSessions=int(total or 0),
        completedSessions=int(completed or 0),
        avgFormScore=float(avg_form) if avg_form is not None else None,
    )


@router.get("/overview", response_model=AnalyticsOverviewResponse)
def analytics_overview(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalyticsOverviewResponse:
    return _overview(db, current_user.id)


@router.get("/exercise/{exercise_id}", response_model=AnalyticsOverviewResponse)
def analytics_exercise(
    exercise_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AnalyticsOverviewResponse:
    return _overview(db, current_user.id, exercise_id)
