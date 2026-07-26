"""User profile endpoints."""

from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .deps import get_current_user
from .schemas import (
    BodyMetricCreateRequest,
    BodyMetricResponse,
    DeleteUserMeRequest,
    UpdateUserMeRequest,
    UserMeResponse,
)
from ..core.security import verify_password
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


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_me(
    payload: DeleteUserMeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Permanently delete the caller's own account and all data derived from it.

    Google Play requires an in-app path to account deletion for any app that
    lets users create an account. Every table referencing ``users.id`` declares
    ON DELETE CASCADE (or SET NULL), so the database removes the dependent rows;
    the relationships use passive_deletes so SQLAlchemy does not try to NULL out
    NOT NULL foreign keys on the way.
    """
    if not verify_password(payload.password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Password is incorrect",
        )

    if current_user.role == "admin":
        # Losing the last admin would leave the platform unadministrable.
        remaining_admins = db.scalar(
            select(func.count())
            .select_from(User)
            .where(User.role == "admin", User.id != current_user.id)
        )
        if not remaining_admins:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete the last admin account",
            )

    db.delete(current_user)
    db.commit()


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

