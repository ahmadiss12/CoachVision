"""Subscription plan endpoints."""

from fastapi import APIRouter, Depends

from .deps import get_entitlements
from .schemas import EntitlementLimits, EntitlementsResponse
from ..core.entitlements import Entitlements

router = APIRouter()


@router.get("/me/entitlements", response_model=EntitlementsResponse)
def my_entitlements(
    entitlements: Entitlements = Depends(get_entitlements),
) -> EntitlementsResponse:
    """The caller's plan and what it allows, for clients to show or lock features."""
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
    )
