"""Database bootstrap helpers for local development."""

from sqlalchemy import inspect, select, text

from ..core.security import hash_password
from .base import Base
from .models import Exercise, User
from .session import engine, SessionLocal

ADMIN_USER_SEED: dict[str, str] = {
    "email": "admin@coachvision.test",
    "password": "Admin1234",
    "display_name": "Admin User",
}

EXERCISE_SEEDS: list[dict[str, str]] = [
    {"id": "squat", "name": "Squat", "description": "Lower-body compound squat movement."},
    {"id": "pushup", "name": "Push-Up", "description": "Upper-body pushing movement."},
    {"id": "lunge", "name": "Lunge", "description": "Single-leg lower-body movement."},
    {"id": "deadlift", "name": "Deadlift", "description": "Hip-hinge posterior chain movement."},
    {"id": "plank", "name": "Plank", "description": "Core isometric hold exercise."},
    {"id": "bicep_curl", "name": "Bicep Curl", "description": "Elbow flexion upper-body movement."},
    {"id": "shoulder_press", "name": "Shoulder Press", "description": "Overhead pressing movement."},
    {"id": "situp", "name": "Sit-Up", "description": "Trunk flexion core exercise."},
    {"id": "jumping_jack", "name": "Jumping Jack", "description": "Full-body cardio movement."},
    {"id": "high_knees", "name": "High Knees", "description": "Cardio drill emphasizing hip flexion."},
    {"id": "mountain_climber", "name": "Mountain Climber", "description": "Dynamic core and cardio movement."},
    {"id": "wall_sit", "name": "Wall Sit", "description": "Isometric lower-body hold."},
]


def _ensure_user_profile_columns() -> None:
    inspector = inspect(engine)
    try:
        existing = {col["name"] for col in inspector.get_columns("users")}
    except Exception:  # noqa: BLE001
        return

    statements: list[str] = []
    if "date_of_birth" not in existing:
        statements.append("ALTER TABLE users ADD COLUMN date_of_birth DATE")
    if "height_cm" not in existing:
        statements.append("ALTER TABLE users ADD COLUMN height_cm NUMERIC(5, 2)")
    if "weight_kg" not in existing:
        statements.append("ALTER TABLE users ADD COLUMN weight_kg NUMERIC(6, 2)")
    if "body_fat_percent" not in existing:
        statements.append("ALTER TABLE users ADD COLUMN body_fat_percent NUMERIC(5, 2)")

    if not statements:
        return

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def bootstrap_database() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_user_profile_columns()

    db = SessionLocal()
    try:
        existing_ids = set(db.scalars(select(Exercise.id)).all())
        for item in EXERCISE_SEEDS:
            if item["id"] in existing_ids:
                continue
            db.add(
                Exercise(
                    id=item["id"],
                    name=item["name"],
                    description=item["description"],
                    default_difficulty="intermediate",
                )
            )

        existing_admin = db.scalar(select(User).where(User.email == ADMIN_USER_SEED["email"]))
        if not existing_admin:
            db.add(
                User(
                    email=ADMIN_USER_SEED["email"],
                    password_hash=hash_password(ADMIN_USER_SEED["password"]),
                    display_name=ADMIN_USER_SEED["display_name"],
                )
            )

        db.commit()
    finally:
        db.close()

