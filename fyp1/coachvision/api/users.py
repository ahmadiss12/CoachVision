"""User profile endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .deps import get_current_user
from .schemas import UpdateUserMeRequest, UserMeResponse
from ..db.models import User
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
    for key, value in data.items():
        setattr(current_user, key, value)
    current_user.updated_at = datetime.now(timezone.utc)
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return UserMeResponse.model_validate(current_user)

