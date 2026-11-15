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
        # The app derives the target from the readiness response, so it is the
        # app that withholds it. Same for the body progress charts, which are
        # drawn from the user's own body metrics.
        "next_session_target",
        "body_progress",
        "analytics",
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

# Applied during early access and to admins: every feature and no limits, which
# is exactly how the app behaved before plans existed (trainers had no client
# or program cap). Early access must not introduce a limit of its own.
UNRESTRICTED = PlanDefinition(
    features=PRO_FEATURES,
    exercise_ids=None,
    history_days=None,
    max_clients=None,
    max_programs=None,
)


@dataclass(frozen=True)
class Entitlements:
    """What one account may do right now.

    ``plan`` is the plan stored on the account. ``effective_plan`` is the plan
    the account is treated as, which is the top plan during early access and
    for admins. In those two cases ``definition`` also carries no limits.
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

    While ``settings.early_access`` is on, nothing is locked or limited for
    anyone, so paid plans can be built before they launch. Admins are never
    locked or limited.
    """
    stored = plan if plan in PLAN_DEFINITIONS else "free"
    early_access = settings.early_access
    unrestricted = early_access or role == "admin"
    effective = PLANS[-1] if unrestricted else stored
    return Entitlements(
        plan=stored,
        effective_plan=effective,
        early_access=early_access,
        definition=UNRESTRICTED if unrestricted else PLAN_DEFINITIONS[effective],
    )


def minimum_plan_for(feature: str) -> str | None:
    """Lowest plan that includes ``feature``, or None if no plan does."""
    for plan in PLANS:
        if feature in PLAN_DEFINITIONS[plan].features:
            return plan
    return None


def minimum_plan_for_exercise(exercise_id: str) -> str:
    """Lowest plan that may start a session for ``exercise_id``."""
    for plan in PLANS:
        allowed = PLAN_DEFINITIONS[plan].exercise_ids
        if allowed is None or exercise_id in allowed:
            return plan
    return PLANS[-1]
