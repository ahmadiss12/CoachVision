"""What each subscription plan may do.

This is the single source of truth for plan features and limits. Routes ask
``resolve_entitlements`` (or the ``require_feature`` dependency) instead of
comparing plan names themselves, so changing what a plan includes is an edit
to this file only.

Plans form a ladder: plus includes everything in free, pro everything in plus.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import settings

PLANS: tuple[str, ...] = ("free", "plus", "pro")

# Exercises a free account can start a live session for. Paid plans get all.
FREE_EXERCISE_IDS: frozenset[str] = frozenset(
    {"squat", "pushup", "lunge", "plank", "jumping_jack", "high_knees"}
)

PLUS_FEATURES: frozenset[str] = frozenset(
    {
        "weighted_training",  # external load is recorded and counts toward fatigue
        "full_session_review",  # every issue and cue, not just the top one
        "full_form_breakdown",  # all squat fault probabilities
        "readiness_detail",  # readiness score, reasons, history
        "next_session_target",
        "analytics",
        "body_progress",  # body-fat logging and progress charts
        "pdf_report",
    }
)

PRO_FEATURES: frozenset[str] = PLUS_FEATURES | frozenset(
    {
        "live_monitoring",  # observe a client's live session
        "full_client_session_detail",
    }
)


@dataclass(frozen=True)
class PlanDefinition:
    """Features and limits of one plan. ``None`` means unlimited."""

    features: frozenset[str]
    exercise_ids: frozenset[str] | None
    history_days: int | None
    max_clients: int | None
    max_programs: int | None


PLAN_DEFINITIONS: dict[str, PlanDefinition] = {
    "free": PlanDefinition(
        features=frozenset(),
        exercise_ids=FREE_EXERCISE_IDS,
        history_days=14,
        max_clients=2,
        max_programs=1,
    ),
    "plus": PlanDefinition(
        features=PLUS_FEATURES,
        exercise_ids=None,
        history_days=None,
        max_clients=2,
        max_programs=1,
    ),
    "pro": PlanDefinition(
        features=PRO_FEATURES,
        exercise_ids=None,
        history_days=None,
        max_clients=30,
        max_programs=None,
    ),
}


@dataclass(frozen=True)
class Entitlements:
    """What one account may do right now.

    ``plan`` is the plan stored on the account. ``effective_plan`` is the plan
    whose features and limits apply, which is higher than ``plan`` during
    early access and for admins.
    """

    plan: str
    effective_plan: str
    early_access: bool
    definition: PlanDefinition

    def has_feature(self, feature: str) -> bool:
        return feature in self.definition.features

    def can_use_exercise(self, exercise_id: str) -> bool:
        allowed = self.definition.exercise_ids
        return allowed is None or exercise_id in allowed


def resolve_entitlements(plan: str, role: str = "client") -> Entitlements:
    """Return the entitlements for an account on ``plan`` with ``role``.

    While ``settings.early_access`` is on, every account gets the top plan so
    nothing is locked before paid plans launch. Admins always get the top plan.
    """
    stored = plan if plan in PLAN_DEFINITIONS else "free"
    early_access = settings.early_access
    effective = PLANS[-1] if early_access or role == "admin" else stored
    return Entitlements(
        plan=stored,
        effective_plan=effective,
        early_access=early_access,
        definition=PLAN_DEFINITIONS[effective],
    )


def minimum_plan_for(feature: str) -> str | None:
    """Lowest plan that includes ``feature``, or None if no plan does."""
    for plan in PLANS:
        if feature in PLAN_DEFINITIONS[plan].features:
            return plan
    return None
