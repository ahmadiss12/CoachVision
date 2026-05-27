"""Workout session lifecycle endpoints."""

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from .deps import get_current_user
from .schemas import CreateSessionRequest, EndSessionRequest, SessionFeedbackResponse, SessionResponse
from ..db.models import Exercise, Session, SessionFeedback, User
from ..db.session import get_db

router = APIRouter(prefix="/sessions")


@router.post("", response_model=SessionResponse)
def create_session(
    payload: CreateSessionRequest,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SessionResponse:
    exercise = db.get(Exercise, payload.exercise_id)
    if not exercise:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exercise not found")

    session = Session(
        user_id=current_user.id,
        exercise_id=payload.exercise_id,
        difficulty=payload.difficulty,
        status="planned",
        target_sets=payload.target_sets,
        target_reps=payload.target_reps,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionResponse.model_validate(session)


@router.get("", response_model=list[SessionResponse])
def list_sessions(
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[SessionResponse]:
    rows = db.scalars(
        select(Session)
        .where(Session.user_id == current_user.id, Session.status == "completed")
        .order_by(Session.ended_at.desc(), Session.created_at.desc())
    ).all()
    return [SessionResponse.model_validate(row) for row in rows]


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: UUID,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SessionResponse:
    session = db.get(Session, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return SessionResponse.model_validate(session)


@router.get("/{session_id}/feedback", response_model=SessionFeedbackResponse)
def get_session_feedback(
    session_id: UUID,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SessionFeedbackResponse:
    session = db.get(Session, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    row = db.scalars(
        select(SessionFeedback).where(
            SessionFeedback.session_id == session_id,
            SessionFeedback.user_id == current_user.id,
        )
    ).first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session feedback not found",
        )
    return SessionFeedbackResponse.model_validate(row)


@router.post("/{session_id}/start", response_model=SessionResponse)
def start_session(
    session_id: UUID,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SessionResponse:
    session = db.get(Session, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    session.status = "active"
    if not session.started_at:
        session.started_at = datetime.now(timezone.utc)
    session.updated_at = datetime.now(timezone.utc)
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionResponse.model_validate(session)


@router.post("/{session_id}/end", response_model=SessionResponse)
def end_session(
    session_id: UUID,
    payload: EndSessionRequest | None = None,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SessionResponse:
    session = db.get(Session, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.status == "completed":
        from ..services.session_feedback_generator import ensure_session_feedback_for_completed_workout

        ensure_session_feedback_for_completed_workout(db, session)
        db.commit()
        db.refresh(session)
        return SessionResponse.model_validate(session)

    session.status = "completed"
    now_utc = datetime.now(timezone.utc)
    session.ended_at = now_utc
    if session.started_at:
        session.duration_seconds = max(0, int((now_utc - session.started_at).total_seconds()))
    if payload is not None and payload.total_reps is not None:
        session.total_reps = payload.total_reps
    if payload is not None and payload.avg_form_score is not None:
        session.avg_form_score = payload.avg_form_score
    session.updated_at = now_utc
    db.add(session)
    from ..services.session_feedback_generator import ensure_session_feedback_for_completed_workout

    ensure_session_feedback_for_completed_workout(db, session)
    db.commit()
    db.refresh(session)
    from ..services.fatigue_post_session import run_after_completed_session

    run_after_completed_session(db, session.id)
    return SessionResponse.model_validate(session)

