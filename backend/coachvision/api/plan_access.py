"""Apply subscription plan limits inside routes.

``core/entitlements.py`` says what each plan allows. The helpers here answer
the questions routes actually ask ("may this user start this exercise?",
"how far back may they see?"), which need the database, and shape the
responses and errors that go with them.

While early access is on every helper takes its unrestricted path and runs no
extra queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session as DbSession

from .schemas import SessionFeedbackResponse
from ..core.entitlements import (
    Entitlements,
    minimum_plan_for_exercise,
    resolve_entitlements,
)
from ..db.models import (
    Program,
    ProgramAssignment,
    ProgramDay,
    ProgramExercise,
    TrainerClient,
    User,
)


def entitlements_for(user: User) -> Entitlements:
    return resolve_entitlements(user.subscription_plan, user.role)


def plan_required(detail: str) -> HTTPException:
    """402 for "your plan does not include this", distinct from a 403 role error."""
    return HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=detail)


@dataclass(frozen=True)
class ExerciseAccess:
    allowed: bool  # may start a session for the exercise
    weighted: bool  # may record external load for it


def _sponsored_exercises(db: DbSession, client_id: UUID) -> dict[str, bool]:
    """Exercises unlocked for a client by a trainer's plan, mapped to whether
    external load is unlocked too.

    The trainer pays, not the client: an exercise in a program that a trainer
    assigned to this client is available when the trainer's own plan includes
    it, even if the client's plan does not. The coaching link and the
    assignment must both still be active.
    """
    rows = db.execute(
        select(ProgramExercise.exercise_id, User.subscription_plan, User.role)
        .select_from(ProgramAssignment)
        .join(Program, Program.id == ProgramAssignment.program_id)
        .join(ProgramDay, ProgramDay.program_id == Program.id)
        .join(ProgramExercise, ProgramExercise.program_day_id == ProgramDay.id)
        .join(User, User.id == ProgramAssignment.trainer_id)
        .join(
            TrainerClient,
            (TrainerClient.trainer_id == ProgramAssignment.trainer_id)
            & (TrainerClient.client_id == ProgramAssignment.client_id),
        )
        .where(
            ProgramAssignment.client_id == client_id,
            ProgramAssignment.status == "active",
            TrainerClient.status == "active",
        )
        .distinct()
    ).all()

    sponsored: dict[str, bool] = {}
    for exercise_id, trainer_plan, trainer_role in rows:
        trainer = resolve_entitlements(trainer_plan, trainer_role)
        if trainer.can_use_exercise(exercise_id):
            weighted = trainer.has_feature("weighted_training")
            sponsored[exercise_id] = sponsored.get(exercise_id, False) or weighted
    return sponsored


def sponsored_exercise_ids(db: DbSession, user: User) -> list[str]:
    """Exercises outside the user's own plan that a trainer's plan unlocks."""
    own = entitlements_for(user)
    if own.definition.exercise_ids is None:
        return []
    return sorted(
        exercise_id
        for exercise_id in _sponsored_exercises(db, user.id)
        if not own.can_use_exercise(exercise_id)
    )


def exercise_access(db: DbSession, user: User, exercise_id: str) -> ExerciseAccess:
    own = entitlements_for(user)
    allowed = own.can_use_exercise(exercise_id)
    weighted = own.has_feature("weighted_training")
    if allowed and weighted:
        return ExerciseAccess(allowed=True, weighted=True)

    sponsored = _sponsored_exercises(db, user.id)
    if exercise_id in sponsored:
        return ExerciseAccess(allowed=True, weighted=weighted or sponsored[exercise_id])
    return ExerciseAccess(allowed=allowed, weighted=weighted)


def require_exercise_access(db: DbSession, user: User, exercise_id: str) -> ExerciseAccess:
    access = exercise_access(db, user, exercise_id)
    if not access.allowed:
        raise plan_required(exercise_locked_message(exercise_id))
    return access


def exercise_locked_message(exercise_id: str) -> str:
    return f"This exercise requires the {minimum_plan_for_exercise(exercise_id)} plan"


def history_cutoff(entitlements: Entitlements, now: datetime | None = None) -> datetime | None:
    """Oldest moment the plan may look back to, or None for unlimited history."""
    days = entitlements.definition.history_days
    if days is None:
        return None
    return (now or datetime.now(timezone.utc)) - timedelta(days=days)


def is_before_cutoff(moment: datetime | None, cutoff: datetime | None) -> bool:
    if moment is None or cutoff is None:
        return False
    if moment.tzinfo is None:  # SQLite returns naive datetimes
        moment = moment.replace(tzinfo=timezone.utc)
    return moment < cutoff


def history_locked_message(entitlements: Entitlements) -> str:
    days = entitlements.definition.history_days
    return f"Sessions older than {days} days require the plus plan"


def trim_feedback(feedback: SessionFeedbackResponse) -> SessionFeedbackResponse:
    """Short review for plans without the full one: the top issue and one cue.

    ``errors_count`` is kept so clients can say how much more there is.
    """
    top = feedback.top_errors[:1]
    kept_codes = {item.code for item in top}
    return feedback.model_copy(
        update={
            "top_errors": top,
            "action_items": feedback.action_items[:1],
            "error_breakdown": {
                code: entry
                for code, entry in feedback.error_breakdown.items()
                if code in kept_codes
            },
            "full_review": False,
        }
    )


def active_client_count(db: DbSession, trainer_id: UUID) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(TrainerClient)
            .where(TrainerClient.trainer_id == trainer_id, TrainerClient.status == "active")
        )
        or 0
    )


def has_client_capacity(db: DbSession, trainer: User) -> bool:
    """Whether the trainer's plan lets them take on one more client.

    A trainer already above the limit keeps every existing client; the limit
    only stops new ones.
    """
    limit = entitlements_for(trainer).definition.max_clients
    return limit is None or active_client_count(db, trainer.id) < limit
