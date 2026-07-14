"""Trainer-side endpoints: invites, client roster, client session history.

Isolation model: every trainer query goes through the trainer_clients link
table, so a trainer can only ever read data for clients with an *active* link
to them. Clients accept invites; they never appear in a roster without one.
"""

import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from .deps import get_current_user, require_role
from .reports import _exercise_name, _rep_stats, _to_float
from .schemas import (
    DailyReportSessionResponse,
    InviteAcceptRequest,
    InviteAcceptResponse,
    InviteCreateRequest,
    InviteResponse,
    SessionFeedbackResponse,
    SessionResponse,
    TrainerClientResponse,
)
from ..db.models import ClientInvite, Exercise, RepEvent, Session, SessionFeedback, TrainerClient, User
from ..db.session import get_db

_HOLD_EXERCISE_IDS = frozenset({"plank", "wall_sit"})

router = APIRouter()

_now = lambda: datetime.now(timezone.utc)  # noqa: E731


def _get_active_link(db: DbSession, trainer_id: UUID, client_id: UUID) -> TrainerClient | None:
    return db.scalar(
        select(TrainerClient).where(
            TrainerClient.trainer_id == trainer_id,
            TrainerClient.client_id == client_id,
            TrainerClient.status == "active",
        )
    )


@router.post(
    "/trainer/invites",
    response_model=InviteResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_invite(
    payload: InviteCreateRequest,
    db: DbSession = Depends(get_db),
    trainer: User = Depends(require_role("trainer")),
) -> InviteResponse:
    invite = ClientInvite(
        trainer_id=trainer.id,
        token=secrets.token_urlsafe(16),
        email=payload.email.lower().strip() if payload.email else None,
        expires_at=_now() + timedelta(days=payload.expires_in_days),
    )
    db.add(invite)
    db.commit()
    db.refresh(invite)
    return InviteResponse.model_validate(invite)


@router.get("/trainer/invites", response_model=list[InviteResponse])
def list_invites(
    db: DbSession = Depends(get_db),
    trainer: User = Depends(require_role("trainer")),
) -> list[InviteResponse]:
    rows = db.scalars(
        select(ClientInvite)
        .where(ClientInvite.trainer_id == trainer.id)
        .order_by(ClientInvite.created_at.desc())
    ).all()
    return [InviteResponse.model_validate(row) for row in rows]


@router.delete("/trainer/invites/{invite_id}", response_model=InviteResponse)
def revoke_invite(
    invite_id: UUID,
    db: DbSession = Depends(get_db),
    trainer: User = Depends(require_role("trainer")),
) -> InviteResponse:
    invite = db.get(ClientInvite, invite_id)
    if not invite or invite.trainer_id != trainer.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite not found")
    if invite.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Invite is {invite.status}, only pending invites can be revoked",
        )
    invite.status = "revoked"
    db.commit()
    db.refresh(invite)
    return InviteResponse.model_validate(invite)


@router.post("/invites/accept", response_model=InviteAcceptResponse)
def accept_invite(
    payload: InviteAcceptRequest,
    db: DbSession = Depends(get_db),
    client: User = Depends(require_role("client")),
) -> InviteAcceptResponse:
    invite = db.scalar(select(ClientInvite).where(ClientInvite.token == payload.token))
    if not invite or invite.status != "pending":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid invite code")

    expires_at = invite.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < _now():
        invite.status = "expired"
        db.commit()
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Invite has expired")

    if invite.email and invite.email != client.email:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This invite was issued for a different email address",
        )

    link = db.scalar(
        select(TrainerClient).where(
            TrainerClient.trainer_id == invite.trainer_id,
            TrainerClient.client_id == client.id,
        )
    )
    if link is None:
        link = TrainerClient(trainer_id=invite.trainer_id, client_id=client.id, status="active")
        db.add(link)
    else:
        link.status = "active"
        link.ended_at = None

    invite.status = "accepted"
    invite.accepted_by = client.id
    invite.accepted_at = _now()
    db.commit()

    trainer = db.get(User, invite.trainer_id)
    return InviteAcceptResponse(
        trainer_id=invite.trainer_id,
        trainer_name=trainer.display_name if trainer else "",
        status="active",
    )


@router.get("/trainer/clients", response_model=list[TrainerClientResponse])
def list_clients(
    db: DbSession = Depends(get_db),
    trainer: User = Depends(require_role("trainer")),
) -> list[TrainerClientResponse]:
    rows = db.execute(
        select(TrainerClient, User)
        .join(User, User.id == TrainerClient.client_id)
        .where(
            TrainerClient.trainer_id == trainer.id,
            TrainerClient.status == "active",
        )
        .order_by(TrainerClient.created_at.desc())
    ).all()
    return [
        TrainerClientResponse(
            client_id=link.client_id,
            display_name=user.display_name,
            email=user.email,
            status=link.status,
            linked_at=link.created_at,
        )
        for link, user in rows
    ]


@router.get("/trainer/clients/{client_id}/sessions", response_model=list[SessionResponse])
def client_sessions(
    client_id: UUID,
    db: DbSession = Depends(get_db),
    trainer: User = Depends(require_role("trainer")),
) -> list[SessionResponse]:
    if _get_active_link(db, trainer.id, client_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")
    rows = db.scalars(
        select(Session)
        .where(Session.user_id == client_id, Session.status == "completed")
        .order_by(Session.ended_at.desc(), Session.created_at.desc())
    ).all()
    return [SessionResponse.model_validate(row) for row in rows]


@router.get(
    "/trainer/clients/{client_id}/sessions/{session_id}",
    response_model=DailyReportSessionResponse,
)
def client_session_detail(
    client_id: UUID,
    session_id: UUID,
    db: DbSession = Depends(get_db),
    trainer: User = Depends(require_role("trainer")),
) -> DailyReportSessionResponse:
    """Full coaching breakdown of one client session.

    Same shape as the client's own daily report entries: targets vs actual,
    volume load, the generated form review (top errors + action items), and
    aggregated per-rep quality stats (ROM, tempo, depth).
    """
    if _get_active_link(db, trainer.id, client_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")

    session = db.get(Session, session_id)
    if not session or session.user_id != client_id or session.status != "completed":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    exercise = db.get(Exercise, session.exercise_id)
    feedback = db.scalar(
        select(SessionFeedback).where(SessionFeedback.session_id == session.id)
    )
    rep_rows = db.scalars(
        select(RepEvent).where(RepEvent.session_id == session.id).order_by(RepEvent.rep_number)
    ).all()

    external_load = _to_float(session.external_load_kg)
    uses_external_load = bool(external_load and external_load > 0)
    estimated_volume_load = None
    if uses_external_load and session.exercise_id not in _HOLD_EXERCISE_IDS and session.total_reps:
        estimated_volume_load = round(external_load * session.total_reps, 2)

    return DailyReportSessionResponse(
        session_id=session.id,
        exercise_id=session.exercise_id,
        exercise_name=_exercise_name(
            session.exercise_id, {exercise.id: exercise} if exercise else {}
        ),
        difficulty=session.difficulty,
        target_sets=session.target_sets,
        target_reps=session.target_reps,
        total_reps=session.total_reps,
        avg_form_score=_to_float(session.avg_form_score),
        duration_seconds=session.duration_seconds,
        started_at=session.started_at,
        ended_at=session.ended_at,
        external_load_kg=external_load,
        body_weight_kg=_to_float(session.body_weight_kg),
        uses_external_load=uses_external_load,
        estimated_volume_load_kg=estimated_volume_load,
        feedback=SessionFeedbackResponse.model_validate(feedback) if feedback else None,
        rep_stats=_rep_stats(rep_rows),
    )


@router.delete("/trainer/clients/{client_id}", status_code=status.HTTP_204_NO_CONTENT)
def end_client_link(
    client_id: UUID,
    db: DbSession = Depends(get_db),
    trainer: User = Depends(require_role("trainer")),
) -> None:
    link = _get_active_link(db, trainer.id, client_id)
    if link is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")
    link.status = "ended"
    link.ended_at = _now()
    db.commit()


@router.get("/me/trainers", response_model=list[TrainerClientResponse])
def my_trainers(
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[TrainerClientResponse]:
    """Client-side view: which trainers currently coach me."""
    rows = db.execute(
        select(TrainerClient, User)
        .join(User, User.id == TrainerClient.trainer_id)
        .where(
            TrainerClient.client_id == current_user.id,
            TrainerClient.status == "active",
        )
        .order_by(TrainerClient.created_at.desc())
    ).all()
    return [
        TrainerClientResponse(
            client_id=link.trainer_id,
            display_name=user.display_name,
            email=user.email,
            status=link.status,
            linked_at=link.created_at,
        )
        for link, user in rows
    ]
