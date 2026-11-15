"""Subscription plan entitlements (GET /v1/me/entitlements and the plan rules).

Covers the plan ladder itself, the early-access switch that unlocks everything
before paid plans launch, and the admin route that sets a plan by hand.
"""

import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from coachvision.api.deps import require_feature
from coachvision.api.schemas import UpdatePlanRequest
from coachvision.core.config import settings
from coachvision.core.entitlements import (
    FREE_EXERCISE_IDS,
    PLAN_DEFINITIONS,
    PLANS,
    minimum_plan_for,
    resolve_entitlements,
)
from coachvision.db.base import Base
from coachvision.db.bootstrap import EXERCISE_SEEDS
from coachvision.db.models import User
from coachvision.db.session import get_db
from coachvision.main import app

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


client = TestClient(app)


def _register(role: str = "client") -> tuple[str, dict[str, str]]:
    email = f"{role}-{uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/v1/auth/register",
        json={"email": email, "password": "secret123", "display_name": role, "role": role},
    )
    assert resp.status_code == 200, resp.text
    return email, {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _user(email: str) -> User:
    db = TestingSessionLocal()
    try:
        return db.scalar(select(User).where(User.email == email))
    finally:
        db.close()


def _set(email: str, **fields) -> None:
    db = TestingSessionLocal()
    user = db.scalar(select(User).where(User.email == email))
    for key, value in fields.items():
        setattr(user, key, value)
    db.commit()
    db.close()


class PlanRulesTest(unittest.TestCase):
    """The plan definitions, with no HTTP or database involved."""

    def test_plans_form_a_ladder(self) -> None:
        for lower, higher in zip(PLANS, PLANS[1:]):
            low, high = PLAN_DEFINITIONS[lower], PLAN_DEFINITIONS[higher]
            self.assertLessEqual(low.features, high.features, f"{higher} must include {lower}")
            for limit in ("history_days", "max_clients", "max_programs"):
                low_value, high_value = getattr(low, limit), getattr(high, limit)
                if low_value is None:
                    self.assertIsNone(high_value, f"{higher}.{limit} is tighter than {lower}")
                elif high_value is not None:
                    self.assertGreaterEqual(high_value, low_value, f"{higher}.{limit}")

    def test_free_exercises_are_real_exercises(self) -> None:
        seeded = {seed["id"] for seed in EXERCISE_SEEDS}
        self.assertLessEqual(FREE_EXERCISE_IDS, seeded)

    def test_free_plan_limits_apply_when_early_access_is_off(self) -> None:
        with patch.object(settings, "early_access", False):
            free = resolve_entitlements("free")
        self.assertEqual(free.effective_plan, "free")
        self.assertTrue(free.can_use_exercise("squat"))
        self.assertFalse(free.can_use_exercise("deadlift"))
        self.assertFalse(free.has_feature("analytics"))
        self.assertEqual(free.definition.history_days, 14)

    def test_plus_unlocks_athlete_features_but_not_coaching(self) -> None:
        with patch.object(settings, "early_access", False):
            plus = resolve_entitlements("plus")
        self.assertTrue(plus.can_use_exercise("deadlift"))
        self.assertTrue(plus.has_feature("analytics"))
        self.assertFalse(plus.has_feature("live_monitoring"))
        self.assertEqual(plus.definition.max_clients, PLAN_DEFINITIONS["free"].max_clients)

    def _assert_unrestricted(self, entitlements) -> None:
        self.assertEqual(entitlements.effective_plan, "pro")
        self.assertTrue(entitlements.has_feature("live_monitoring"))
        self.assertTrue(entitlements.can_use_exercise("deadlift"))
        for limit in ("exercise_ids", "history_days", "max_clients", "max_programs"):
            self.assertIsNone(getattr(entitlements.definition, limit), limit)

    def test_early_access_locks_and_limits_nothing(self) -> None:
        # Trainers had no client or program cap before plans existed, so early
        # access must not hand them the pro caps either.
        for plan in PLANS:
            with patch.object(settings, "early_access", True):
                entitlements = resolve_entitlements(plan, role="trainer")
            self.assertEqual(entitlements.plan, plan)
            self.assertTrue(entitlements.early_access)
            self._assert_unrestricted(entitlements)

    def test_admin_is_never_locked_or_limited(self) -> None:
        with patch.object(settings, "early_access", False):
            admin = resolve_entitlements("free", role="admin")
        self.assertFalse(admin.early_access)
        self._assert_unrestricted(admin)

    def test_plan_names_agree_everywhere(self) -> None:
        # The plan list lives in the rules, the database enum and the admin
        # request schema. They must not drift apart.
        self.assertEqual(tuple(PLAN_DEFINITIONS), PLANS)
        self.assertEqual(tuple(User.__table__.c.subscription_plan.type.enums), PLANS)
        for plan in PLANS:
            self.assertEqual(UpdatePlanRequest(plan=plan).plan, plan)

    def test_unknown_stored_plan_falls_back_to_free(self) -> None:
        with patch.object(settings, "early_access", False):
            self.assertEqual(resolve_entitlements("gold").effective_plan, "free")

    def test_minimum_plan_for_feature(self) -> None:
        self.assertEqual(minimum_plan_for("analytics"), "plus")
        self.assertEqual(minimum_plan_for("live_monitoring"), "pro")
        self.assertIsNone(minimum_plan_for("no_such_feature"))

    def test_require_feature_rejects_with_402_and_names_the_plan(self) -> None:
        check = require_feature("analytics")
        with patch.object(settings, "early_access", False):
            with self.assertRaises(HTTPException) as raised:
                check(resolve_entitlements("free"))
            self.assertEqual(raised.exception.status_code, 402)
            self.assertIn("plus", raised.exception.detail)
            self.assertTrue(check(resolve_entitlements("plus")).has_feature("analytics"))


class EntitlementsApiTest(unittest.TestCase):
    _previous_override = None

    @classmethod
    def setUpClass(cls) -> None:
        # Scoped to this class: other test modules install their own override.
        cls._previous_override = app.dependency_overrides.get(get_db)
        app.dependency_overrides[get_db] = _override_get_db
        Base.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._previous_override is None:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = cls._previous_override

    def test_requires_login(self) -> None:
        self.assertEqual(client.get("/v1/me/entitlements").status_code, 401)

    def test_new_account_starts_on_free(self) -> None:
        email, _ = _register()
        self.assertEqual(_user(email).subscription_plan, "free")

    def test_early_access_reports_full_access_on_a_free_account(self) -> None:
        _, headers = _register()
        with patch.object(settings, "early_access", True):
            body = client.get("/v1/me/entitlements", headers=headers).json()
        self.assertEqual(body["plan"], "free")
        self.assertEqual(body["effectivePlan"], "pro")
        self.assertTrue(body["earlyAccess"])
        self.assertIn("analytics", body["features"])
        self.assertEqual(
            body["limits"],
            {"exerciseIds": None, "historyDays": None, "maxClients": None, "maxPrograms": None},
        )

    def test_free_account_sees_free_limits_after_early_access(self) -> None:
        _, headers = _register()
        with patch.object(settings, "early_access", False):
            body = client.get("/v1/me/entitlements", headers=headers).json()
        self.assertEqual(body["effectivePlan"], "free")
        self.assertFalse(body["earlyAccess"])
        self.assertEqual(body["features"], [])
        self.assertEqual(
            body["limits"],
            {
                "exerciseIds": sorted(FREE_EXERCISE_IDS),
                "historyDays": 14,
                "maxClients": 2,
                "maxPrograms": 1,
            },
        )

    def test_user_cannot_choose_their_own_plan(self) -> None:
        # A plan must only change through the admin route (and later, billing).
        smuggled = {"subscription_plan": "pro", "subscriptionPlan": "pro", "plan": "pro"}
        email = f"smuggle-{uuid4().hex[:8]}@example.com"
        resp = client.post(
            "/v1/auth/register",
            json={"email": email, "password": "secret123", "display_name": "x", **smuggled},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(_user(email).subscription_plan, "free")

        headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
        resp = client.patch("/v1/users/me", json={"display_name": "y", **smuggled}, headers=headers)
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(_user(email).display_name, "y")
        self.assertEqual(_user(email).subscription_plan, "free")

    def test_admin_sets_plan_and_entitlements_follow(self) -> None:
        admin_email, admin_headers = _register()
        _set(admin_email, role="admin")
        trainer_email, trainer_headers = _register("trainer")
        trainer_id = str(_user(trainer_email).id)

        resp = client.patch(
            f"/v1/admin/users/{trainer_id}/plan", json={"plan": "pro"}, headers=admin_headers
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["subscriptionPlan"], "pro")

        with patch.object(settings, "early_access", False):
            body = client.get("/v1/me/entitlements", headers=trainer_headers).json()
        self.assertEqual(body["plan"], "pro")
        self.assertEqual(body["limits"]["maxClients"], 30)
        self.assertIn("live_monitoring", body["features"])

    def test_plan_route_is_admin_only_and_validates(self) -> None:
        email, headers = _register()
        user_id = str(_user(email).id)
        resp = client.patch(f"/v1/admin/users/{user_id}/plan", json={"plan": "pro"}, headers=headers)
        self.assertEqual(resp.status_code, 403, resp.text)
        self.assertEqual(_user(email).subscription_plan, "free")

        admin_email, admin_headers = _register()
        _set(admin_email, role="admin")
        bad = client.patch(
            f"/v1/admin/users/{user_id}/plan", json={"plan": "gold"}, headers=admin_headers
        )
        self.assertEqual(bad.status_code, 422, bad.text)
        missing = client.patch(
            f"/v1/admin/users/{uuid4()}/plan", json={"plan": "plus"}, headers=admin_headers
        )
        self.assertEqual(missing.status_code, 404, missing.text)


if __name__ == "__main__":
    unittest.main()
