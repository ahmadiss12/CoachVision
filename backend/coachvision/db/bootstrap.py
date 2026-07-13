"""Database bootstrap helpers for local development."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import select

from ..core.security import hash_password
from .models import Exercise, User
from .session import SessionLocal

BACKEND_ROOT = Path(__file__).resolve().parents[2]

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


def run_migrations() -> None:
    """Apply Alembic migrations up to head."""
    alembic_cfg = Config(str(BACKEND_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    command.upgrade(alembic_cfg, "head")


def bootstrap_database() -> None:
    run_migrations()

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

