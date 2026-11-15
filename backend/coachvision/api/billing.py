"""Subscription plan endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .deps import get_current_user
from .plan_access import entitlements_for, sponsored_exercise_ids
from .schemas import EntitlementLimits, EntitlementsResponse
from ..db.models import User
from ..db.session import get_db

router = APIRouter()


@router.get("/me/entitlements", response_model=EntitlementsResponse)
def my_entitlements(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EntitlementsResponse:
    """The caller's plan and what it allows, for clients to show or lock features."""
    entitlements = entitlements_for(current_user)
    definition = entitlements.definition
    return EntitlementsResponse(
        plan=entitlements.plan,
        effective_plan=entitlements.effective_plan,
        early_access=entitlements.early_access,
        features=sorted(definition.features),
        limits=EntitlementLimits(
            exercise_ids=None if definition.exercise_ids is None else sorted(definition.exercise_ids),
            history_days=definition.history_days,
            max_clients=definition.max_clients,
            max_programs=definition.max_programs,
        ),
        sponsored_exercise_ids=sponsored_exercise_ids(db, current_user),
    )
