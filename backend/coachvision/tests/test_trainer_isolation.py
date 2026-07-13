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
