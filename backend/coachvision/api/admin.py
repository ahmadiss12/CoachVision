"""Admin endpoints: user listing and role management."""

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from .deps import require_role
from .schemas import AdminUserResponse, UpdateRoleRequest
from ..db.models import TrainerClient, User
from ..db.session import get_db

router = APIRouter(prefix="/admin")


@router.get("/users", response_model=list[AdminUserResponse])
def list_users(
    db: DbSession = Depends(get_db),
    admin: User = Depends(require_role("admin")),
) -> list[AdminUserResponse]:
    rows = db.scalars(select(User).order_by(User.created_at.desc())).all()
    return [AdminUserResponse.model_validate(row) for row in rows]


@router.patch("/users/{user_id}/role", response_model=AdminUserResponse)
def change_role(
    user_id: UUID,
    payload: UpdateRoleRequest,
    db: DbSession = Depends(get_db),
    admin: User = Depends(require_role("admin")),
) -> AdminUserResponse:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if user.id == admin.id and payload.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot remove your own admin role",
        )

    old_role = user.role
    user.role = payload.role
    user.updated_at = datetime.now(timezone.utc)

    # A demoted trainer must not keep read access to client data.
    if old_role == "trainer" and payload.role != "trainer":
        links = db.scalars(
            select(TrainerClient).where(
                TrainerClient.trainer_id == user.id,
                TrainerClient.status == "active",
            )
        ).all()
        for link in links:
            link.status = "ended"
            link.ended_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(user)
    return AdminUserResponse.model_validate(user)
