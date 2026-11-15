"""Plan limits enforced by the API once early access is switched off.

Every test here runs with ``settings.early_access`` off, except the last class,
which proves the same requests are all allowed while it is on.
"""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.websockets import WebSocketDisconnect

from coachvision.core.config import settings
from coachvision.db.base import Base
from coachvision.db.bootstrap import EXERCISE_SEEDS
from coachvision.db.models import (
    ClientInvite,
    Exercise,
    FatiguePrediction,
    Session as WorkoutSession,
    SessionFeedback,
    TrainerClient,
    User,
)
from coachvision.db.session import get_db
from coachvision.main import app
from coachvision.realtime.connection_manager import live_registry

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

ISSUES = ("squat_shallow_depth", "squat_forward_lean", "squat_knee_valgus")


class Account:
    """A registered user: auth headers plus the raw token for WebSockets."""

    def __init__(self, role: str = "client", plan: str = "free") -> None:
        self.email = f"{role}-{uuid4().hex[:10]}@example.com"
        resp = client.post(
            "/v1/auth/register",
            json={"email": self.email, "password": "secret123", "display_name": role, "role": role},
        )
        assert resp.status_code == 200, resp.text
        self.token = resp.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
        db = TestingSessionLocal()
        user = db.scalar(select(User).where(User.email == self.email))
        self.id = user.id
        user.subscription_plan = plan
        db.commit()
        db.close()

    def set(self, **fields) -> None:
        db = TestingSessionLocal()
        user = db.get(User, self.id)
        for key, value in fields.items():
            setattr(user, key, value)
        db.commit()
        db.close()


def _completed_session(user: Account, *, days_ago: float, exercise_id: str = "squat") -> str:
    """A finished workout with a three-issue review, ended ``days_ago`` days back."""
    ended = datetime.now(timezone.utc) - timedelta(days=days_ago)
    db = TestingSessionLocal()
    session = WorkoutSession(
        user_id=user.id,
        exercise_id=exercise_id,
        difficulty="intermediate",
        status="completed",
        target_sets=1,
        target_reps=10,
        total_reps=10,
        avg_form_score=70,
        started_at=ended - timedelta(minutes=5),
        ended_at=ended,
    )
    db.add(session)
    db.flush()
    db.add(
        SessionFeedback(
            user_id=user.id,
            exercise_id=exercise_id,
            session_id=session.id,
            overall_rating=70,
            summary_text="Completed 10 rep(s).",
            errors_count=6,
            top_errors=[
                {"code": code, "label": code, "count": 3 - i, "severity": "medium"}
                for i, code in enumerate(ISSUES)
            ],
            error_breakdown={
                code: {"count": 3 - i, "severity": "medium", "why": "w", "fix": "f"}
                for i, code in enumerate(ISSUES)
            },
            action_items=[
                {"priority": i + 1, "title": code, "why": "w", "how_to_fix": "f", "cue": "c"}
                for i, code in enumerate(ISSUES)
            ],
        )
    )
    db.commit()
    session_id = str(session.id)
    db.close()
    return session_id


def _link(trainer: Account, client_account: Account):
    """Invite + accept. Returns the accept response."""
    invite = client.post("/v1/trainer/invites", json={}, headers=trainer.headers)
    assert invite.status_code == 201, invite.text
    return client.post(
        "/v1/invites/accept", json={"token": invite.json()["token"]}, headers=client_account.headers
    )


def _program_body(*exercise_ids: str, load: float | None = None) -> dict:
    return {
        "name": f"Program {uuid4().hex[:6]}",
        "weeks": 1,
        "days": [
            {
                "dayIndex": 1,
                "exercises": [
                    {"exerciseId": ex, "targetSets": 3, "targetReps": 8, "externalLoadKg": load}
                    for ex in exercise_ids
                ],
            }
        ],
    }


def _new_session(user: Account, exercise_id: str, **extra):
    return client.post(
        "/v1/sessions", json={"exerciseId": exercise_id, **extra}, headers=user.headers
    )


def _live_socket(user: Account):
    return client.websocket_connect(f"/v1/ws/live?token={user.token}")


def _ws_start(user: Account, exercise: str, **extra) -> dict:
    """Open the live socket, send start, return the server's first reply."""
    with _live_socket(user) as ws:
        ws.send_json({"type": "start", "exerciseName": exercise, **extra})
        return ws.receive_json()


class _PlanLimitsCase(unittest.TestCase):
    early_access = False
    _previous_override = None

    @classmethod
    def setUpClass(cls) -> None:
        # Scoped to the class: other test modules install their own override.
        cls._previous_override = app.dependency_overrides.get(get_db)
        app.dependency_overrides[get_db] = _override_get_db
        Base.metadata.create_all(bind=engine)
        db = TestingSessionLocal()
        for seed in EXERCISE_SEEDS:
            if db.get(Exercise, seed["id"]) is None:
                db.add(Exercise(**seed))
        db.commit()
        db.close()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._previous_override is None:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = cls._previous_override

    def setUp(self) -> None:
        patcher = patch.object(settings, "early_access", self.early_access)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Closing a live socket mid-workout marks the session aborted, which
        # subtracts datetimes. SQLite hands them back without a timezone
        # (Postgres does not), so skip that cleanup here.
        abort = patch("coachvision.api.ws_live.abort_active_workout")
        abort.start()
        self.addCleanup(abort.stop)


class ExerciseLimitsTest(_PlanLimitsCase):
    def test_free_starts_free_exercises_only(self) -> None:
        free = Account()
        self.assertEqual(_new_session(free, "squat").status_code, 200)
        locked = _new_session(free, "deadlift")
        self.assertEqual(locked.status_code, 402, locked.text)
        self.assertIn("plus", locked.json()["detail"])

    def test_plus_starts_every_exercise(self) -> None:
        plus = Account(plan="plus")
        for seed in EXERCISE_SEEDS:
            self.assertEqual(_new_session(plus, seed["id"]).status_code, 200, seed["id"])

    def test_unknown_exercise_is_still_a_404(self) -> None:
        self.assertEqual(_new_session(Account(), "handstand").status_code, 404)

    def test_external_load_needs_weighted_training(self) -> None:
        free, plus = Account(), Account(plan="plus")
        self.assertIsNone(_new_session(free, "squat", externalLoadKg=40).json()["externalLoadKg"])
        self.assertEqual(_new_session(plus, "squat", externalLoadKg=40).json()["externalLoadKg"], 40)

    def test_load_sent_when_ending_a_session_is_ignored_on_free(self) -> None:
        for account, expected in ((Account(), None), (Account(plan="plus"), 25)):
            session_id = _new_session(account, "squat").json()["id"]
            ended = client.post(
                f"/v1/sessions/{session_id}/end",
                json={"totalReps": 5, "externalLoadKg": 25},
                headers=account.headers,
            )
            self.assertEqual(ended.status_code, 200, ended.text)
            self.assertEqual(ended.json()["externalLoadKg"], expected)
            self.assertEqual(ended.json()["totalReps"], 5)

    def test_live_socket_rejects_a_locked_exercise(self) -> None:
        free = Account()
        reply = _ws_start(free, "deadlift")
        self.assertEqual(reply["type"], "error")
        self.assertEqual(reply["code"], "PLAN_REQUIRED")
        db = TestingSessionLocal()
        rows = db.scalars(select(WorkoutSession).where(WorkoutSession.user_id == free.id)).all()
        db.close()
        self.assertEqual(rows, [], "a rejected start must not create a session")

    def test_live_socket_starts_allowed_exercises(self) -> None:
        self.assertEqual(_ws_start(Account(), "squat")["type"], "started")
        self.assertEqual(_ws_start(Account(plan="plus"), "deadlift")["type"], "started")

    def test_live_socket_drops_load_on_free(self) -> None:
        for account, expected in ((Account(), None), (Account(plan="plus"), 30.0)):
            reply = _ws_start(account, "squat", externalLoadKg=30)
            self.assertEqual(reply["type"], "started")
            db = TestingSessionLocal()
            row = db.get(WorkoutSession, UUID(reply["sessionId"]))
            load = None if row.external_load_kg is None else float(row.external_load_kg)
            db.close()
            self.assertEqual(load, expected)

    def test_owner_gets_fault_probabilities_only_with_the_full_breakdown(self) -> None:
        fake = {
            "kind": "metrics",
            "session_id": "x",
            "count": 1,
            "state": "up",
            "angle": 170,
            "feedback": "",
            "progress": 0,
            "form_name": "forward_lean",
            "form_probabilities": {"good": 0.2, "forward_lean": 0.8},
            "confidence": 0.9,
        }
        for account, expected in (
            (Account(), None),
            (Account(plan="plus"), fake["form_probabilities"]),
        ):
            with patch("coachvision.api.ws_live.process_pose_landmarks", return_value=fake):
                with _live_socket(account) as ws:
                    ws.send_json({"type": "start", "exerciseName": "squat"})
                    session_id = ws.receive_json()["sessionId"]
                    ws.send_json({"type": "pose", "sessionId": session_id, "landmarks": []})
                    metrics = ws.receive_json()
            self.assertEqual(metrics["type"], "metrics")
            self.assertEqual(metrics["formName"], "forward_lean", "the top fault stays visible")
            self.assertEqual(metrics["formProbabilities"], expected)


class TrainerSponsoredExerciseTest(_PlanLimitsCase):
    """The trainer pays, not the client."""

    def _assigned(self, trainer: Account, athlete: Account, *exercises: str, load=None) -> None:
        self.assertEqual(_link(trainer, athlete).status_code, 200)
        program = client.post(
            "/v1/trainer/programs", json=_program_body(*exercises, load=load), headers=trainer.headers
        )
        self.assertEqual(program.status_code, 201, program.text)
        assigned = client.post(
            f"/v1/trainer/programs/{program.json()['id']}/assign",
            json={"clientId": str(athlete.id)},
            headers=trainer.headers,
        )
        self.assertEqual(assigned.status_code, 200, assigned.text)

    def test_pro_trainers_program_unlocks_its_exercises_for_a_free_client(self) -> None:
        trainer, athlete = Account("trainer", "pro"), Account()
        self._assigned(trainer, athlete, "deadlift", load=60)

        created = _new_session(athlete, "deadlift", externalLoadKg=60)
        self.assertEqual(created.status_code, 200, created.text)
        self.assertEqual(created.json()["externalLoadKg"], 60, "the trainer's plan covers load too")
        self.assertEqual(_ws_start(athlete, "deadlift")["type"], "started")

        # Only what the program prescribes, not every paid exercise.
        self.assertEqual(_new_session(athlete, "bicep_curl").status_code, 402)

        entitlements = client.get("/v1/me/entitlements", headers=athlete.headers).json()
        self.assertEqual(entitlements["plan"], "free")
        self.assertEqual(entitlements["sponsoredExerciseIds"], ["deadlift"])
        self.assertNotIn("deadlift", entitlements["limits"]["exerciseIds"])

    def test_unlock_ends_with_the_coaching_link(self) -> None:
        trainer, athlete = Account("trainer", "pro"), Account()
        self._assigned(trainer, athlete, "deadlift")
        self.assertEqual(_new_session(athlete, "deadlift").status_code, 200)

        ended = client.delete(f"/v1/trainer/clients/{athlete.id}", headers=trainer.headers)
        self.assertEqual(ended.status_code, 204, ended.text)
        self.assertEqual(_new_session(athlete, "deadlift").status_code, 402)
        self.assertEqual(
            client.get("/v1/me/entitlements", headers=athlete.headers).json()["sponsoredExerciseIds"],
            [],
        )

    def test_unlock_ends_when_the_assignment_is_cancelled(self) -> None:
        trainer, athlete = Account("trainer", "pro"), Account()
        self._assigned(trainer, athlete, "deadlift")
        assignments = client.get(
            f"/v1/trainer/clients/{athlete.id}/assignments", headers=trainer.headers
        ).json()
        active = [row for row in assignments if row["status"] == "active"]
        cancelled = client.delete(
            f"/v1/trainer/assignments/{active[0]['id']}", headers=trainer.headers
        )
        self.assertEqual(cancelled.status_code, 204, cancelled.text)
        self.assertEqual(_new_session(athlete, "deadlift").status_code, 402)

    def test_unlock_ends_when_the_trainer_downgrades(self) -> None:
        trainer, athlete = Account("trainer", "pro"), Account()
        self._assigned(trainer, athlete, "deadlift")
        trainer.set(subscription_plan="free")
        self.assertEqual(_new_session(athlete, "deadlift").status_code, 402)

    def test_another_clients_assignment_unlocks_nothing(self) -> None:
        trainer, athlete, stranger = Account("trainer", "pro"), Account(), Account()
        self._assigned(trainer, athlete, "deadlift")
        self.assertEqual(_new_session(stranger, "deadlift").status_code, 402)

    def test_free_trainer_cannot_prescribe_locked_exercises_or_load(self) -> None:
        trainer = Account("trainer")
        locked = client.post(
            "/v1/trainer/programs", json=_program_body("deadlift"), headers=trainer.headers
        )
        self.assertEqual(locked.status_code, 402, locked.text)

        allowed = client.post(
            "/v1/trainer/programs", json=_program_body("squat", load=50), headers=trainer.headers
        )
        self.assertEqual(allowed.status_code, 201, allowed.text)
        exercise = allowed.json()["days"][0]["exercises"][0]
        self.assertIsNone(exercise["externalLoadKg"])


class HistoryAndReviewLimitsTest(_PlanLimitsCase):
    def test_free_history_is_the_last_14_days(self) -> None:
        free = Account()
        recent = _completed_session(free, days_ago=13)
        old = _completed_session(free, days_ago=15)

        listed = [row["id"] for row in client.get("/v1/sessions", headers=free.headers).json()]
        self.assertEqual(listed, [recent])

        self.assertEqual(client.get(f"/v1/sessions/{recent}", headers=free.headers).status_code, 200)
        for path in (f"/v1/sessions/{old}", f"/v1/sessions/{old}/feedback"):
            resp = client.get(path, headers=free.headers)
            self.assertEqual(resp.status_code, 402, path)
            self.assertIn("14 days", resp.json()["detail"])

    def test_locked_history_comes_back_on_upgrade(self) -> None:
        user = Account()
        old = _completed_session(user, days_ago=200)
        self.assertEqual(client.get("/v1/sessions", headers=user.headers).json(), [])

        user.set(subscription_plan="plus")
        listed = [row["id"] for row in client.get("/v1/sessions", headers=user.headers).json()]
        self.assertEqual(listed, [old])
        self.assertEqual(client.get(f"/v1/sessions/{old}/feedback", headers=user.headers).status_code, 200)

    def test_unfinished_sessions_are_never_history_locked(self) -> None:
        free = Account()
        session_id = _new_session(free, "squat").json()["id"]
        self.assertEqual(client.get(f"/v1/sessions/{session_id}", headers=free.headers).status_code, 200)

    def test_free_review_is_the_top_issue_and_one_cue(self) -> None:
        free = Account()
        session_id = _completed_session(free, days_ago=1)
        body = client.get(f"/v1/sessions/{session_id}/feedback", headers=free.headers).json()

        self.assertFalse(body["fullReview"])
        self.assertEqual([item["code"] for item in body["topErrors"]], [ISSUES[0]])
        self.assertEqual(list(body["errorBreakdown"]), [ISSUES[0]])
        self.assertEqual(len(body["actionItems"]), 1)
        self.assertEqual(body["errorsCount"], 6, "the total stays, so the app can show what is locked")
        self.assertEqual(body["overallRating"], 70)

    def test_plus_review_is_complete(self) -> None:
        plus = Account(plan="plus")
        session_id = _completed_session(plus, days_ago=1)
        body = client.get(f"/v1/sessions/{session_id}/feedback", headers=plus.headers).json()
        self.assertTrue(body["fullReview"])
        self.assertEqual(len(body["topErrors"]), 3)
        self.assertEqual(len(body["errorBreakdown"]), 3)
        self.assertEqual(len(body["actionItems"]), 3)

    def test_trimming_does_not_change_what_is_stored(self) -> None:
        user = Account()
        session_id = _completed_session(user, days_ago=1)
        client.get(f"/v1/sessions/{session_id}/feedback", headers=user.headers)
        db = TestingSessionLocal()
        stored = db.scalar(select(SessionFeedback).where(SessionFeedback.user_id == user.id))
        self.assertEqual(len(stored.top_errors), 3)
        db.close()


class ReadinessAndReportLimitsTest(_PlanLimitsCase):
    def _predict(self, account: Account) -> dict:
        resp = client.post(
            "/v1/fatigue/predict",
            json={"exerciseId": "squat", "userContext": {"sleepHours": 4, "muscleSoreness": 5}},
            headers=account.headers,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def test_free_readiness_is_the_fatigue_level_only(self) -> None:
        free = Account()
        body = self._predict(free)
        self.assertTrue(body["detailLocked"])
        self.assertIn(body["fatigueLevel"], ("low", "moderate", "high"))
        self.assertEqual(body["factors"], [])
        self.assertEqual(body["explainability"], [])
        self.assertEqual(body["featureSnapshot"], {})
        self.assertEqual(body["recommendation"], "")

        history = client.get("/v1/fatigue/history?exerciseId=squat", headers=free.headers)
        self.assertEqual(history.status_code, 402, history.text)

    def test_free_prediction_is_stored_in_full_for_a_later_upgrade(self) -> None:
        user = Account()
        self._predict(user)
        db = TestingSessionLocal()
        stored = db.scalar(select(FatiguePrediction).where(FatiguePrediction.user_id == user.id))
        self.assertTrue(stored.recommendation)
        self.assertTrue(stored.input_snapshot["explainability"])
        db.close()

        user.set(subscription_plan="plus")
        history = client.get("/v1/fatigue/history?exerciseId=squat", headers=user.headers).json()
        self.assertEqual(len(history), 1)
        self.assertTrue(history[0]["explainability"])
        self.assertFalse(history[0]["detailLocked"])

    def test_plus_readiness_is_complete(self) -> None:
        body = self._predict(Account(plan="plus"))
        self.assertFalse(body["detailLocked"])
        self.assertTrue(body["recommendation"])
        self.assertTrue(body["explainability"])

    def test_analytics_and_daily_report_need_plus(self) -> None:
        free, plus = Account(), Account(plan="plus")
        for path in ("/v1/analytics/overview", "/v1/analytics/exercise/squat", "/v1/reports/daily"):
            self.assertEqual(client.get(path, headers=free.headers).status_code, 402, path)
            self.assertEqual(client.get(path, headers=plus.headers).status_code, 200, path)

    def test_plan_checks_do_not_replace_login(self) -> None:
        for path in ("/v1/analytics/overview", "/v1/reports/daily", "/v1/fatigue/history?exerciseId=squat"):
            self.assertEqual(client.get(path).status_code, 401, path)


class TrainerLimitsTest(_PlanLimitsCase):
    def test_free_trainer_is_limited_to_two_clients(self) -> None:
        trainer = Account("trainer")
        for _ in range(2):
            self.assertEqual(_link(trainer, Account()).status_code, 200)

        blocked = client.post("/v1/trainer/invites", json={}, headers=trainer.headers)
        self.assertEqual(blocked.status_code, 402, blocked.text)
        self.assertIn("2 active clients", blocked.json()["detail"])

        trainer.set(subscription_plan="pro")
        self.assertEqual(_link(trainer, Account()).status_code, 200)

    def test_plus_does_not_raise_the_client_limit(self) -> None:
        trainer = Account("trainer", "plus")
        for _ in range(2):
            self.assertEqual(_link(trainer, Account()).status_code, 200)
        self.assertEqual(
            client.post("/v1/trainer/invites", json={}, headers=trainer.headers).status_code, 402
        )

    def test_invite_accepted_after_the_trainer_filled_up_stays_pending(self) -> None:
        trainer = Account("trainer")
        tokens = [
            client.post("/v1/trainer/invites", json={}, headers=trainer.headers).json()["token"]
            for _ in range(3)
        ]
        late = Account()
        for token in tokens[:2]:
            ok = client.post("/v1/invites/accept", json={"token": token}, headers=Account().headers)
            self.assertEqual(ok.status_code, 200, ok.text)

        full = client.post("/v1/invites/accept", json={"token": tokens[2]}, headers=late.headers)
        self.assertEqual(full.status_code, 409, full.text)
        db = TestingSessionLocal()
        invite = db.scalar(select(ClientInvite).where(ClientInvite.token == tokens[2]))
        self.assertEqual(invite.status, "pending")
        db.close()

        trainer.set(subscription_plan="pro")
        retry = client.post("/v1/invites/accept", json={"token": tokens[2]}, headers=late.headers)
        self.assertEqual(retry.status_code, 200, retry.text)

    def test_trainer_over_the_limit_keeps_existing_clients(self) -> None:
        trainer = Account("trainer", "pro")
        clients = [Account() for _ in range(3)]
        for account in clients:
            self.assertEqual(_link(trainer, account).status_code, 200)

        trainer.set(subscription_plan="free")
        roster = client.get("/v1/trainer/clients", headers=trainer.headers).json()
        self.assertEqual(len(roster), 3, "a downgrade never removes clients")
        self.assertEqual(
            client.post("/v1/trainer/invites", json={}, headers=trainer.headers).status_code, 402
        )
        db = TestingSessionLocal()
        active = db.scalars(
            select(TrainerClient).where(
                TrainerClient.trainer_id == trainer.id, TrainerClient.status == "active"
            )
        ).all()
        self.assertEqual(len(active), 3)
        db.close()

    def test_free_trainer_is_limited_to_one_program(self) -> None:
        trainer = Account("trainer")
        first = client.post("/v1/trainer/programs", json=_program_body("squat"), headers=trainer.headers)
        self.assertEqual(first.status_code, 201, first.text)
        second = client.post("/v1/trainer/programs", json=_program_body("squat"), headers=trainer.headers)
        self.assertEqual(second.status_code, 402, second.text)

        # Deleting frees the slot; pro has no limit.
        client.delete(f"/v1/trainer/programs/{first.json()['id']}", headers=trainer.headers)
        again = client.post("/v1/trainer/programs", json=_program_body("squat"), headers=trainer.headers)
        self.assertEqual(again.status_code, 201, again.text)
        trainer.set(subscription_plan="pro")
        for _ in range(3):
            resp = client.post(
                "/v1/trainer/programs", json=_program_body("deadlift"), headers=trainer.headers
            )
            self.assertEqual(resp.status_code, 201, resp.text)

    def test_live_monitoring_needs_pro(self) -> None:
        trainer, athlete = Account("trainer"), Account()
        self.assertEqual(_link(trainer, athlete).status_code, 200)
        session_id = f"live-{uuid4().hex[:8]}"
        live_registry.register_session(session_id, str(athlete.id), "squat")
        self.addCleanup(live_registry.unregister_session, session_id)

        self.assertEqual(client.get("/v1/trainer/clients/live", headers=trainer.headers).json(), [])
        with client.websocket_connect(f"/v1/ws/observe/{session_id}?token={trainer.token}") as ws:
            reply = ws.receive_json()
            self.assertEqual(reply["code"], "PLAN_REQUIRED")
            with self.assertRaises(WebSocketDisconnect) as closed:
                ws.receive_json()
            self.assertEqual(closed.exception.code, 4402)
        self.assertEqual(live_registry.observer_count(session_id), 0)

        trainer.set(subscription_plan="pro")
        live = client.get("/v1/trainer/clients/live", headers=trainer.headers).json()
        self.assertEqual([row["sessionId"] for row in live], [session_id])
        with client.websocket_connect(f"/v1/ws/observe/{session_id}?token={trainer.token}") as ws:
            self.assertEqual(ws.receive_json()["type"], "observing")

    def test_client_session_detail_follows_the_trainers_plan(self) -> None:
        trainer, athlete = Account("trainer"), Account(plan="plus")
        self.assertEqual(_link(trainer, athlete).status_code, 200)
        recent = _completed_session(athlete, days_ago=1)
        old = _completed_session(athlete, days_ago=40)
        detail = f"/v1/trainer/clients/{athlete.id}/sessions"

        # Free trainer: summary only, and the trainer's own 14-day window,
        # even though the client is on plus.
        listed = client.get(detail, headers=trainer.headers).json()
        self.assertEqual([row["id"] for row in listed], [recent])
        self.assertEqual(client.get(f"{detail}/{old}", headers=trainer.headers).status_code, 402)
        summary = client.get(f"{detail}/{recent}", headers=trainer.headers).json()
        self.assertFalse(summary["feedback"]["fullReview"])
        self.assertEqual(len(summary["feedback"]["topErrors"]), 1)
        self.assertEqual(summary["totalReps"], 10)

        # Pro trainer: everything, even for a free client.
        trainer.set(subscription_plan="pro")
        athlete.set(subscription_plan="free")
        self.assertEqual(len(client.get(detail, headers=trainer.headers).json()), 2)
        full = client.get(f"{detail}/{old}", headers=trainer.headers).json()
        self.assertTrue(full["feedback"]["fullReview"])
        self.assertEqual(len(full["feedback"]["topErrors"]), 3)

    def test_chat_and_coaching_basics_stay_free(self) -> None:
        trainer, athlete = Account("trainer"), Account()
        self.assertEqual(_link(trainer, athlete).status_code, 200)
        sent = client.post(
            f"/v1/chat/{trainer.id}/messages", json={"body": "hi coach"}, headers=athlete.headers
        )
        self.assertIn(sent.status_code, (200, 201), sent.text)
        self.assertEqual(client.get("/v1/me/trainers", headers=athlete.headers).status_code, 200)
        self.assertEqual(client.get("/v1/me/plan/today", headers=athlete.headers).status_code, 200)

    def test_admin_is_never_limited(self) -> None:
        admin = Account()
        admin.set(role="admin")
        self.assertEqual(_new_session(admin, "deadlift").status_code, 200)
        self.assertEqual(client.get("/v1/analytics/overview", headers=admin.headers).status_code, 200)


class EarlyAccessUnlocksEverythingTest(_PlanLimitsCase):
    """The same requests the classes above see refused, on a free account."""

    early_access = True

    def test_nothing_is_locked_for_a_free_athlete(self) -> None:
        free = Account()
        created = _new_session(free, "deadlift", externalLoadKg=80)
        self.assertEqual(created.status_code, 200, created.text)
        self.assertEqual(created.json()["externalLoadKg"], 80)
        self.assertEqual(_ws_start(free, "deadlift")["type"], "started")

        old = _completed_session(free, days_ago=300)
        self.assertIn(old, [row["id"] for row in client.get("/v1/sessions", headers=free.headers).json()])
        review = client.get(f"/v1/sessions/{old}/feedback", headers=free.headers).json()
        self.assertTrue(review["fullReview"])
        self.assertEqual(len(review["topErrors"]), 3)

        for path in (
            "/v1/analytics/overview",
            "/v1/reports/daily",
            "/v1/fatigue/history?exerciseId=squat",
        ):
            self.assertEqual(client.get(path, headers=free.headers).status_code, 200, path)
        predicted = client.post(
            "/v1/fatigue/predict", json={"exerciseId": "squat"}, headers=free.headers
        ).json()
        self.assertFalse(predicted["detailLocked"])

    def test_nothing_is_limited_for_a_free_trainer(self) -> None:
        trainer = Account("trainer")
        clients = [Account() for _ in range(4)]
        for account in clients:
            self.assertEqual(_link(trainer, account).status_code, 200)
        for _ in range(3):
            resp = client.post(
                "/v1/trainer/programs", json=_program_body("deadlift", load=70), headers=trainer.headers
            )
            self.assertEqual(resp.status_code, 201, resp.text)
            self.assertEqual(resp.json()["days"][0]["exercises"][0]["externalLoadKg"], 70)

        session_id = f"live-{uuid4().hex[:8]}"
        live_registry.register_session(session_id, str(clients[0].id), "squat")
        self.addCleanup(live_registry.unregister_session, session_id)
        self.assertEqual(len(client.get("/v1/trainer/clients/live", headers=trainer.headers).json()), 1)
        with client.websocket_connect(f"/v1/ws/observe/{session_id}?token={trainer.token}") as ws:
            self.assertEqual(ws.receive_json()["type"], "observing")


if __name__ == "__main__":
    unittest.main()
