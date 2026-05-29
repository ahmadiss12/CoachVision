"""User profile endpoints."""

from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .deps import get_current_user
from .schemas import BodyMetricCreateRequest, BodyMetricResponse, UpdateUserMeRequest, UserMeResponse
from ..db.models import BodyMetric, User
from ..db.session import get_db

router = APIRouter(prefix="/users")


@router.get("/me", response_model=UserMeResponse)
def get_me(current_user: User = Depends(get_current_user)) -> UserMeResponse:
    return UserMeResponse.model_validate(current_user)


@router.patch("/me", response_model=UserMeResponse)
def update_me(
    payload: UpdateUserMeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UserMeResponse:
    data = payload.model_dump(exclude_none=True)
    if "date_of_birth" in data:
        try:
            data["date_of_birth"] = date.fromisoformat(data["date_of_birth"])
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="date_of_birth must be YYYY-MM-DD",
            ) from exc
    for key, value in data.items():
        setattr(current_user, key, value)
    current_user.updated_at = datetime.now(timezone.utc)
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return UserMeResponse.model_validate(current_user)


@router.get("/me/body-metrics", response_model=list[BodyMetricResponse])
def list_body_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[BodyMetricResponse]:
    items = db.scalars(
        select(BodyMetric)
        .where(BodyMetric.user_id == current_user.id)
        .order_by(BodyMetric.entry_date.desc(), BodyMetric.created_at.desc())
    ).all()
    return [BodyMetricResponse.model_validate(item) for item in items]


@router.post("/me/body-metrics", response_model=BodyMetricResponse, status_code=status.HTTP_201_CREATED)
def create_body_metric(
    payload: BodyMetricCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> BodyMetricResponse:
    try:
        entry_date = date.fromisoformat(payload.date)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="date must be YYYY-MM-DD",
        ) from exc

    metric = BodyMetric(
        user_id=current_user.id,
        entry_date=entry_date,
        weight_kg=payload.weight_kg,
        body_fat_percent=payload.body_fat_percent,
    )
    current_user.weight_kg = payload.weight_kg
    current_user.body_fat_percent = payload.body_fat_percent
    current_user.updated_at = datetime.now(timezone.utc)
    db.add(metric)
    db.add(current_user)
    db.commit()
    db.refresh(metric)
    return BodyMetricResponse.model_validate(metric)

