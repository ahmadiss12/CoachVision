"""Multi-tenancy isolation tests for the trainer/client API (Phase 1).

Uses an in-memory SQLite database and FastAPI dependency overrides so no
Postgres instance (and no Alembic run) is needed.
"""

import unittest
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from coachvision.db.base import Base
from coachvision.db.models import ClientInvite, Exercise, Session as WorkoutSession, User
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


app.dependency_overrides[get_db] = _override_get_db
# Plain TestClient (no context manager) so the lifespan/migration hook never runs.
client = TestClient(app)


def _register(email: str, role: str) -> dict[str, str]:
    resp = client.post(
        "/v1/auth/register",
        json={
            "email": email,
            "password": "secret123",
            "display_name": email.split("@")[0],
            "role": role,
        },
    )
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TrainerIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        Base.metadata.create_all(bind=engine)
        db = TestingSessionLocal()
        db.add(Exercise(id="squat", name="Squat", default_difficulty="intermediate"))
        db.commit()
        db.close()

        cls.trainer_a = _register("trainer.a@test.dev", "trainer")
        cls.trainer_b = _register("trainer.b@test.dev", "trainer")
        cls.client_c = _register("client.c@test.dev", "client")
        cls.client_d = _register("client.d@test.dev", "client")

    def _link_c_to_a(self) -> None:
        invite = client.post(
            "/v1/trainer/invites", json={}, headers=self.trainer_a
        ).json()
        resp = client.post(
            "/v1/invites/accept", json={"token": invite["token"]}, headers=self.client_c
        )
        self.assertEqual(resp.status_code, 200, resp.text)

    def test_01_client_cannot_call_trainer_endpoints(self) -> None:
        resp = client.get("/v1/trainer/clients", headers=self.client_c)
        self.assertEqual(resp.status_code, 403)
        resp = client.post("/v1/trainer/invites", json={}, headers=self.client_c)
        self.assertEqual(resp.status_code, 403)

    def test_02_trainer_cannot_accept_invites(self) -> None:
        resp = client.post(
            "/v1/invites/accept", json={"token": "whatever"}, headers=self.trainer_b
        )
        self.assertEqual(resp.status_code, 403)

    def test_03_invite_accept_links_client(self) -> None:
        self._link_c_to_a()
        roster = client.get("/v1/trainer/clients", headers=self.trainer_a).json()
        self.assertEqual(len(roster), 1)
        self.assertEqual(roster[0]["email"], "client.c@test.dev")

    def test_04_other_trainer_sees_empty_roster(self) -> None:
        self._link_c_to_a()
        roster = client.get("/v1/trainer/clients", headers=self.trainer_b).json()
        self.assertEqual(roster, [])

    def test_05_invite_is_single_use(self) -> None:
        invite = client.post(
            "/v1/trainer/invites", json={}, headers=self.trainer_a
        ).json()
        first = client.post(
            "/v1/invites/accept", json={"token": invite["token"]}, headers=self.client_c
        )
        self.assertEqual(first.status_code, 200)
        second = client.post(
            "/v1/invites/accept", json={"token": invite["token"]}, headers=self.client_d
        )
        self.assertEqual(second.status_code, 404)

    def test_06_expired_invite_rejected(self) -> None:
        db = TestingSessionLocal()
        trainer_a_row = db.query(User).filter_by(email="trainer.a@test.dev").one()
        db.add(
            ClientInvite(
                trainer_id=trainer_a_row.id,
                token="expired-token-123",
                expires_at=datetime.now(timezone.utc) - timedelta(days=1),
            )
        )
        db.commit()
        db.close()

        resp = client.post(
            "/v1/invites/accept", json={"token": "expired-token-123"}, headers=self.client_d
        )
        self.assertEqual(resp.status_code, 410)

    def test_07_trainer_sees_only_linked_client_sessions(self) -> None:
        self._link_c_to_a()

        db = TestingSessionLocal()
        client_c_row = db.query(User).filter_by(email="client.c@test.dev").one()
        db.add(
            WorkoutSession(
                user_id=client_c_row.id,
                exercise_id="squat",
                difficulty="intermediate",
                status="completed",
                total_reps=10,
                ended_at=datetime.now(timezone.utc),
            )
        )
        db.commit()
        client_c_id = str(client_c_row.id)
        db.close()

        linked = client.get(
            f"/v1/trainer/clients/{client_c_id}/sessions", headers=self.trainer_a
        )
        self.assertEqual(linked.status_code, 200)
        self.assertGreaterEqual(len(linked.json()), 1)

        unlinked = client.get(
            f"/v1/trainer/clients/{client_c_id}/sessions", headers=self.trainer_b
        )
        self.assertEqual(unlinked.status_code, 404)

    def test_07b_session_detail_requires_active_link(self) -> None:
        self._link_c_to_a()

        db = TestingSessionLocal()
        client_c_row = db.query(User).filter_by(email="client.c@test.dev").one()
        session_row = WorkoutSession(
            user_id=client_c_row.id,
            exercise_id="squat",
            difficulty="intermediate",
            status="completed",
            total_reps=8,
            target_sets=2,
            target_reps=5,
            ended_at=datetime.now(timezone.utc),
        )
        db.add(session_row)
        db.commit()
        client_c_id = str(client_c_row.id)
        session_id = str(session_row.id)
        db.close()

        detail = client.get(
            f"/v1/trainer/clients/{client_c_id}/sessions/{session_id}", headers=self.trainer_a
        )
        self.assertEqual(detail.status_code, 200, detail.text)
        body = detail.json()
        self.assertEqual(body["totalReps"], 8)
        self.assertEqual(body["targetSets"], 2)
        self.assertIn("repStats", body)

        denied = client.get(
            f"/v1/trainer/clients/{client_c_id}/sessions/{session_id}", headers=self.trainer_b
        )
        self.assertEqual(denied.status_code, 404)

    def test_08_ended_link_hides_client(self) -> None:
        self._link_c_to_a()
        db = TestingSessionLocal()
        client_c_row = db.query(User).filter_by(email="client.c@test.dev").one()
        client_c_id = str(client_c_row.id)
        db.close()

        resp = client.delete(f"/v1/trainer/clients/{client_c_id}", headers=self.trainer_a)
        self.assertEqual(resp.status_code, 204)

        roster = client.get("/v1/trainer/clients", headers=self.trainer_a).json()
        self.assertEqual(roster, [])
        sessions = client.get(
            f"/v1/trainer/clients/{client_c_id}/sessions", headers=self.trainer_a
        )
        self.assertEqual(sessions.status_code, 404)

    def test_14_program_assign_and_today_plan(self) -> None:
        self._link_c_to_a()

        created = client.post(
            "/v1/trainer/programs",
            json={
                "name": "Starter strength",
                "weeks": 1,
                "days": [
                    {
                        "dayIndex": 1,
                        "exercises": [
                            {"exerciseId": "squat", "targetSets": 3, "targetReps": 10}
                        ],
                    }
                ],
            },
            headers=self.trainer_a,
        )
        self.assertEqual(created.status_code, 201, created.text)
        program_id = created.json()["id"]

        db = TestingSessionLocal()
        client_c_id = str(db.query(User).filter_by(email="client.c@test.dev").one().id)
        db.close()

        # Trainer B owns neither the program nor the client link.
        denied = client.post(
            f"/v1/trainer/programs/{program_id}/assign",
            json={"clientId": client_c_id},
            headers=self.trainer_b,
        )
        self.assertEqual(denied.status_code, 404)

        assigned = client.post(
            f"/v1/trainer/programs/{program_id}/assign",
            json={"clientId": client_c_id},
            headers=self.trainer_a,
        )
        self.assertEqual(assigned.status_code, 200, assigned.text)
        self.assertEqual(assigned.json()["status"], "active")

        plan = client.get("/v1/me/plan/today", headers=self.client_c)
        self.assertEqual(plan.status_code, 200, plan.text)
        body = plan.json()
        self.assertTrue(body["hasPlan"])
        self.assertFalse(body["restDay"])
        self.assertEqual(body["dayIndex"], 1)
        self.assertEqual(len(body["exercises"]), 1)
        item = body["exercises"][0]
        self.assertEqual(item["exerciseId"], "squat")
        self.assertEqual(item["prescribedReps"], 10)
        self.assertLessEqual(item["adjustedReps"], 10)
        self.assertIn(item["fatigueLevel"], {"low", "moderate", "high"})

    def test_15_rest_day_when_no_day_defined(self) -> None:
        self._link_c_to_a()
        created = client.post(
            "/v1/trainer/programs",
            json={
                "name": "Day-2 only",
                "weeks": 1,
                "days": [
                    {
                        "dayIndex": 2,
                        "exercises": [
                            {"exerciseId": "squat", "targetSets": 2, "targetReps": 8}
                        ],
                    }
                ],
            },
            headers=self.trainer_a,
        )
        program_id = created.json()["id"]

        db = TestingSessionLocal()
        client_c_id = str(db.query(User).filter_by(email="client.c@test.dev").one().id)
        db.close()

        client.post(
            f"/v1/trainer/programs/{program_id}/assign",
            json={"clientId": client_c_id},
            headers=self.trainer_a,
        )
        plan = client.get("/v1/me/plan/today", headers=self.client_c)
        body = plan.json()
        self.assertTrue(body["hasPlan"])
        self.assertTrue(body["restDay"])
        self.assertEqual(body["dayIndex"], 1)

    def test_16_no_plan_message(self) -> None:
        plan = client.get("/v1/me/plan/today", headers=self.client_d)
        body = plan.json()
        self.assertFalse(body["hasPlan"])

    def test_17_live_sessions_visible_only_to_linked_trainer(self) -> None:
        from coachvision.realtime.connection_manager import live_registry

        self._link_c_to_a()
        db = TestingSessionLocal()
        client_c_id = str(db.query(User).filter_by(email="client.c@test.dev").one().id)
        db.close()

        live_registry.register_session("live-test-session", client_c_id, "squat")
        try:
            mine = client.get("/v1/trainer/clients/live", headers=self.trainer_a)
            self.assertEqual(mine.status_code, 200, mine.text)
            body = mine.json()
            self.assertEqual(len(body), 1)
            self.assertEqual(body[0]["sessionId"], "live-test-session")
            self.assertEqual(body[0]["exerciseId"], "squat")

            others = client.get("/v1/trainer/clients/live", headers=self.trainer_b)
            self.assertEqual(others.json(), [])
        finally:
            live_registry.unregister_session("live-test-session")

    def test_18_chat_requires_active_link(self) -> None:
        self._link_c_to_a()

        db = TestingSessionLocal()
        client_c_id = str(db.query(User).filter_by(email="client.c@test.dev").one().id)
        trainer_a_id = str(db.query(User).filter_by(email="trainer.a@test.dev").one().id)
        db.close()

        # Linked pair can message each other.
        sent = client.post(
            f"/v1/chat/{client_c_id}/messages",
            json={"body": "How was the squat session?"},
            headers=self.trainer_a,
        )
        self.assertEqual(sent.status_code, 201, sent.text)

        reply = client.post(
            f"/v1/chat/{trainer_a_id}/messages",
            json={"body": "Felt strong, thanks coach!"},
            headers=self.client_c,
        )
        self.assertEqual(reply.status_code, 201, reply.text)

        # Unlinked trainer B can neither send nor read.
        denied_send = client.post(
            f"/v1/chat/{client_c_id}/messages",
            json={"body": "hello"},
            headers=self.trainer_b,
        )
        self.assertEqual(denied_send.status_code, 404)
        denied_read = client.get(f"/v1/chat/{client_c_id}/messages", headers=self.trainer_b)
        self.assertEqual(denied_read.status_code, 404)

        # History is visible to both participants, in chronological order.
        history = client.get(f"/v1/chat/{client_c_id}/messages", headers=self.trainer_a)
        self.assertEqual(history.status_code, 200)
        bodies = [m["body"] for m in history.json()]
        self.assertIn("How was the squat session?", bodies)
        self.assertIn("Felt strong, thanks coach!", bodies)

    def test_19_chat_unread_and_conversations(self) -> None:
        self._link_c_to_a()

        db = TestingSessionLocal()
        client_c_id = str(db.query(User).filter_by(email="client.c@test.dev").one().id)
        db.close()

        client.post(
            f"/v1/chat/{client_c_id}/messages",
            json={"body": "New program tomorrow"},
            headers=self.trainer_a,
        )

        unread = client.get("/v1/chat/unread-count", headers=self.client_c).json()
        self.assertGreaterEqual(unread["unread"], 1)

        conversations = client.get("/v1/chat/conversations", headers=self.client_c).json()
        self.assertGreaterEqual(len(conversations), 1)
        trainer_conv = next(c for c in conversations if c["partnerRole"] == "trainer")
        self.assertEqual(trainer_conv["lastMessage"], "New program tomorrow")
        self.assertGreaterEqual(trainer_conv["unreadCount"], 1)

        # Reading the thread clears the unread counter.
        client.get(f"/v1/chat/{trainer_conv['partnerId']}/messages", headers=self.client_c)
        unread_after = client.get("/v1/chat/unread-count", headers=self.client_c).json()
        self.assertEqual(unread_after["unread"], 0)

    def test_09_role_visible_in_me(self) -> None:
        me = client.get("/v1/users/me", headers=self.trainer_a).json()
        self.assertEqual(me["role"], "trainer")
        me = client.get("/v1/users/me", headers=self.client_c).json()
        self.assertEqual(me["role"], "client")

    def test_10_admin_cannot_be_self_registered(self) -> None:
        resp = client.post(
            "/v1/auth/register",
            json={
                "email": "sneaky@test.dev",
                "password": "secret123",
                "display_name": "sneaky",
                "role": "admin",
            },
        )
        self.assertEqual(resp.status_code, 422)

    def test_11_only_admin_can_change_roles(self) -> None:
        # Promote a user directly in the DB to act as admin.
        from coachvision.core.security import hash_password

        db = TestingSessionLocal()
        admin_row = User(
            email="admin@test.dev",
            password_hash=hash_password("secret123"),
            display_name="admin",
            role="admin",
        )
        db.add(admin_row)
        db.commit()
        client_d_row = db.query(User).filter_by(email="client.d@test.dev").one()
        client_d_id = str(client_d_row.id)
        db.close()

        login = client.post(
            "/v1/auth/login", json={"email": "admin@test.dev", "password": "secret123"}
        )
        admin_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        # Non-admins are rejected.
        resp = client.patch(
            f"/v1/admin/users/{client_d_id}/role",
            json={"role": "trainer"},
            headers=self.trainer_a,
        )
        self.assertEqual(resp.status_code, 403)

        # Admin can promote client D to trainer and back.
        resp = client.patch(
            f"/v1/admin/users/{client_d_id}/role",
            json={"role": "trainer"},
            headers=admin_headers,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["role"], "trainer")

        resp = client.patch(
            f"/v1/admin/users/{client_d_id}/role",
            json={"role": "client"},
            headers=admin_headers,
        )
        self.assertEqual(resp.json()["role"], "client")

    def test_13_admin_stats_and_delete(self) -> None:
        from coachvision.core.security import hash_password

        db = TestingSessionLocal()
        if db.query(User).filter_by(email="admin3@test.dev").one_or_none() is None:
            db.add(
                User(
                    email="admin3@test.dev",
                    password_hash=hash_password("secret123"),
                    display_name="admin3",
                    role="admin",
                )
            )
            db.commit()
        admin_id = str(db.query(User).filter_by(email="admin3@test.dev").one().id)
        db.close()

        login = client.post(
            "/v1/auth/login", json={"email": "admin3@test.dev", "password": "secret123"}
        )
        admin_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        # Stats are admin-only and return the expected shape.
        denied = client.get("/v1/admin/stats", headers=self.trainer_a)
        self.assertEqual(denied.status_code, 403)
        stats = client.get("/v1/admin/stats", headers=admin_headers)
        self.assertEqual(stats.status_code, 200, stats.text)
        body = stats.json()
        self.assertGreaterEqual(body["totalUsers"], 4)
        self.assertIn("completedSessions", body)

        # Admin cannot delete themselves.
        resp = client.delete(f"/v1/admin/users/{admin_id}", headers=admin_headers)
        self.assertEqual(resp.status_code, 400)

        # Admin can delete another user; their login stops working.
        victim_headers = _register("victim@test.dev", "client")
        victim_id = client.get("/v1/users/me", headers=victim_headers).json()["id"]
        resp = client.delete(f"/v1/admin/users/{victim_id}", headers=admin_headers)
        self.assertEqual(resp.status_code, 204)
        me = client.get("/v1/users/me", headers=victim_headers)
        self.assertEqual(me.status_code, 401)

    def test_12_demoted_trainer_loses_client_links(self) -> None:
        from coachvision.core.security import hash_password

        self._link_c_to_a()

        db = TestingSessionLocal()
        admin_row = db.query(User).filter_by(email="admin2@test.dev").one_or_none()
        if admin_row is None:
            db.add(
                User(
                    email="admin2@test.dev",
                    password_hash=hash_password("secret123"),
                    display_name="admin2",
                    role="admin",
                )
            )
            db.commit()
        trainer_a_row = db.query(User).filter_by(email="trainer.a@test.dev").one()
        trainer_a_id = str(trainer_a_row.id)
        db.close()

        login = client.post(
            "/v1/auth/login", json={"email": "admin2@test.dev", "password": "secret123"}
        )
        admin_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        resp = client.patch(
            f"/v1/admin/users/{trainer_a_id}/role",
            json={"role": "client"},
            headers=admin_headers,
        )
        self.assertEqual(resp.status_code, 200, resp.text)

        # Restore trainer role, then confirm the old links stayed ended.
        client.patch(
            f"/v1/admin/users/{trainer_a_id}/role",
            json={"role": "trainer"},
            headers=admin_headers,
        )
        roster = client.get("/v1/trainer/clients", headers=self.trainer_a).json()
        self.assertEqual(roster, [])


if __name__ == "__main__":
    unittest.main()
