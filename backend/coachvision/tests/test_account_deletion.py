"""Self-service account deletion (DELETE /v1/users/me).

Google Play requires an in-app account deletion path for any app that allows
account creation, so these tests cover both the happy path and the guards.

Uses in-memory SQLite with foreign keys enforced, so the ON DELETE CASCADE
rules behave like Postgres. Without the PRAGMA, SQLite silently ignores
foreign keys and the cascade assertions would pass for the wrong reason.
"""

import unittest
from datetime import date, datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from coachvision.db.base import Base
from coachvision.db.models import (
    BodyMetric,
    Exercise,
    Program,
    ProgramAssignment,
    Session as WorkoutSession,
    SessionFeedback,
    TrainerClient,
    User,
)
from coachvision.db.session import get_db
from coachvision.main import app

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


client = TestClient(app)

PASSWORD = "secret123"


class _IsolatedDbMixin:
    """Bind get_db to this module's engine only while these tests run.

    Other test modules install their own override at import time, and the last
    import wins for the whole process. Scoping it per class keeps the suite
    order-independent instead of silently querying another module's database.
    """

    _previous_override = None

    @classmethod
    def _install_override(cls) -> None:
        cls._previous_override = app.dependency_overrides.get(get_db)
        app.dependency_overrides[get_db] = _override_get_db
        Base.metadata.create_all(bind=engine)

    @classmethod
    def _restore_override(cls) -> None:
        if cls._previous_override is None:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = cls._previous_override


def _register(email: str, role: str = "client") -> dict[str, str]:
    resp = client.post(
        "/v1/auth/register",
        json={
            "email": email,
            "password": PASSWORD,
            "display_name": email.split("@")[0],
            "role": role,
        },
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _delete_me(headers: dict[str, str], password: str):
    # DELETE with a body: httpx's .delete() has no json= kwarg, so use .request().
    return client.request(
        "DELETE", "/v1/users/me", json={"password": password}, headers=headers
    )


def _user_by_email(db, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email))


class AccountDeletionTest(_IsolatedDbMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._install_override()
        db = TestingSessionLocal()
        if not db.get(Exercise, "squat"):
            db.add(Exercise(id="squat", name="Squat", default_difficulty="beginner"))
            db.commit()
        db.close()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._restore_override()

    def test_deletes_account_and_all_derived_data(self) -> None:
        email = f"del-{uuid4().hex[:8]}@example.com"
        headers = _register(email)

        db = TestingSessionLocal()
        user = _user_by_email(db, email)
        user_id = user.id
        workout = WorkoutSession(
            id=uuid4(),
            user_id=user_id,
            exercise_id="squat",
            difficulty="beginner",
            status="completed",
            total_reps=12,
        )
        db.add(workout)
        db.add(
            BodyMetric(
                id=uuid4(),
                user_id=user_id,
                entry_date=date.today(),
                weight_kg=70,
                body_fat_percent=18,
            )
        )
        db.commit()
        db.add(
            SessionFeedback(
                id=uuid4(),
                user_id=user_id,
                exercise_id="squat",
                session_id=workout.id,
                overall_rating=80,
                summary_text="good work",
                errors_count=0,
                generated_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        )
        db.commit()
        db.close()

        resp = _delete_me(headers, PASSWORD)
        self.assertEqual(resp.status_code, 204, resp.text)

        db = TestingSessionLocal()
        self.assertIsNone(db.get(User, user_id), "user row should be gone")
        for model in (WorkoutSession, BodyMetric, SessionFeedback):
            remaining = db.scalar(
                select(func.count()).select_from(model).where(model.user_id == user_id)
            )
            self.assertEqual(remaining, 0, f"{model.__name__} rows should cascade")
        db.close()

    def test_wrong_password_is_rejected_and_account_survives(self) -> None:
        email = f"wrongpw-{uuid4().hex[:8]}@example.com"
        headers = _register(email)

        resp = _delete_me(headers, "not-my-password")
        self.assertEqual(resp.status_code, 403, resp.text)

        db = TestingSessionLocal()
        self.assertIsNotNone(_user_by_email(db, email), "account must survive")
        db.close()

    def test_requires_authentication(self) -> None:
        resp = client.request("DELETE", "/v1/users/me", json={"password": PASSWORD})
        self.assertIn(resp.status_code, (401, 403), resp.text)

    def test_password_is_required_by_schema(self) -> None:
        email = f"nopw-{uuid4().hex[:8]}@example.com"
        headers = _register(email)
        resp = client.request("DELETE", "/v1/users/me", json={}, headers=headers)
        self.assertEqual(resp.status_code, 422, resp.text)

    def test_last_admin_cannot_delete_itself(self) -> None:
        db = TestingSessionLocal()
        existing_admins = db.scalars(select(User).where(User.role == "admin")).all()
        for admin in existing_admins:
            db.delete(admin)
        db.commit()
        db.close()

        email = f"admin-{uuid4().hex[:8]}@example.com"
        headers = _register(email)
        db = TestingSessionLocal()
        admin_user = _user_by_email(db, email)
        admin_user.role = "admin"
        db.commit()
        db.close()

        resp = _delete_me(headers, PASSWORD)
        self.assertEqual(resp.status_code, 409, resp.text)

        db = TestingSessionLocal()
        self.assertIsNotNone(_user_by_email(db, email), "last admin must survive")
        db.close()

    def test_admin_can_delete_itself_when_another_admin_exists(self) -> None:
        first_email = f"admin1-{uuid4().hex[:8]}@example.com"
        second_email = f"admin2-{uuid4().hex[:8]}@example.com"
        _register(first_email)
        second_headers = _register(second_email)

        db = TestingSessionLocal()
        for email in (first_email, second_email):
            _user_by_email(db, email).role = "admin"
        db.commit()
        db.close()

        resp = _delete_me(second_headers, PASSWORD)
        self.assertEqual(resp.status_code, 204, resp.text)

        db = TestingSessionLocal()
        self.assertIsNone(_user_by_email(db, second_email))
        self.assertIsNotNone(_user_by_email(db, first_email), "other admin untouched")
        db.close()

    def test_deleting_trainer_preserves_client_history(self) -> None:
        """A trainer leaving must not erase their clients' own workout data."""
        trainer_email = f"trainer-{uuid4().hex[:8]}@example.com"
        client_email = f"client-{uuid4().hex[:8]}@example.com"
        trainer_headers = _register(trainer_email, role="trainer")
        _register(client_email, role="client")

        db = TestingSessionLocal()
        trainer = _user_by_email(db, trainer_email)
        athlete = _user_by_email(db, client_email)
        trainer_id, athlete_id = trainer.id, athlete.id

        db.add(TrainerClient(id=uuid4(), trainer_id=trainer_id, client_id=athlete_id))
        program = Program(id=uuid4(), trainer_id=trainer_id, name="Leg Day", weeks=1)
        db.add(program)
        db.commit()

        assignment = ProgramAssignment(
            id=uuid4(),
            program_id=program.id,
            trainer_id=trainer_id,
            client_id=athlete_id,
            start_date=date.today(),
        )
        db.add(assignment)
        db.commit()

        athlete_session = WorkoutSession(
            id=uuid4(),
            user_id=athlete_id,
            exercise_id="squat",
            difficulty="beginner",
            status="completed",
            assignment_id=assignment.id,
            total_reps=10,
        )
        db.add(athlete_session)
        db.commit()
        session_id = athlete_session.id
        db.close()

        resp = _delete_me(trainer_headers, PASSWORD)
        self.assertEqual(resp.status_code, 204, resp.text)

        db = TestingSessionLocal()
        self.assertIsNone(db.get(User, trainer_id), "trainer removed")
        self.assertIsNotNone(db.get(User, athlete_id), "client must survive")

        surviving = db.get(WorkoutSession, session_id)
        self.assertIsNotNone(surviving, "client's workout history must survive")
        self.assertIsNone(
            surviving.assignment_id,
            "assignment link should be SET NULL, not cascade-delete the session",
        )
        links = db.scalar(
            select(func.count())
            .select_from(TrainerClient)
            .where(TrainerClient.trainer_id == trainer_id)
        )
        self.assertEqual(links, 0, "coaching links should be removed")
        db.close()


class AdminDeleteUserRegressionTest(_IsolatedDbMixin, unittest.TestCase):
    """Admin deletion used to fail with IntegrityError once a user had sessions.

    User.sessions had no passive_deletes, so SQLAlchemy tried to NULL out
    sessions.user_id (NOT NULL) rather than letting ON DELETE CASCADE run.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._install_override()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._restore_override()

    def test_orm_delete_of_user_with_sessions_succeeds(self) -> None:
        db = TestingSessionLocal()
        if not db.get(Exercise, "squat"):
            db.add(Exercise(id="squat", name="Squat", default_difficulty="beginner"))
            db.commit()

        user = User(
            id=uuid4(),
            email=f"orm-{uuid4().hex[:8]}@example.com",
            password_hash="x",
            display_name="ORM User",
            role="client",
        )
        db.add(user)
        db.commit()
        user_id = user.id

        db.add(
            WorkoutSession(
                id=uuid4(),
                user_id=user_id,
                exercise_id="squat",
                difficulty="beginner",
                status="completed",
            )
        )
        db.commit()

        db.delete(user)
        db.commit()

        self.assertIsNone(db.get(User, user_id))
        remaining = db.scalar(
            select(func.count())
            .select_from(WorkoutSession)
            .where(WorkoutSession.user_id == user_id)
        )
        self.assertEqual(remaining, 0)
        db.close()


if __name__ == "__main__":
    unittest.main()
