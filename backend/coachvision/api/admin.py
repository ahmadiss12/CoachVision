"""Admin endpoints: user listing, role management, stats, and user removal."""

from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from .deps import require_role
from .schemas import AdminStatsResponse, AdminUserResponse, UpdateRoleRequest
from ..db.models import ClientInvite, Session, TrainerClient, User
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


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: UUID,
    db: DbSession = Depends(get_db),
    admin: User = Depends(require_role("admin")),
) -> None:
    """Permanently remove a user and all their data (sessions, metrics, links)."""
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own account",
        )
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    db.delete(user)
    db.commit()


@router.get("/stats", response_model=AdminStatsResponse)
def platform_stats(
    db: DbSession = Depends(get_db),
    admin: User = Depends(require_role("admin")),
) -> AdminStatsResponse:
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)

    role_counts = dict(
        db.execute(select(User.role, func.count(User.id)).group_by(User.role)).all()
    )
    total_sessions, sessions_last_7d = db.execute(
        select(
            func.count(Session.id).filter(Session.status == "completed"),
            func.count(Session.id).filter(
                Session.status == "completed", Session.ended_at >= week_ago
            ),
        )
    ).one()
    active_links = db.scalar(
        select(func.count(TrainerClient.id)).where(TrainerClient.status == "active")
    )
    pending_invites = db.scalar(
        select(func.count(ClientInvite.id)).where(ClientInvite.status == "pending")
    )

    return AdminStatsResponse(
        total_users=sum(role_counts.values()),
        clients=role_counts.get("client", 0),
        trainers=role_counts.get("trainer", 0),
        admins=role_counts.get("admin", 0),
        completed_sessions=int(total_sessions or 0),
        sessions_last_7d=int(sessions_last_7d or 0),
        active_coaching_links=int(active_links or 0),
        pending_invites=int(pending_invites or 0),
    )
