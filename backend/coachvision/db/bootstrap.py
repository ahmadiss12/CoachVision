"""Database bootstrap helpers for local development."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import select

from ..core.config import settings
from ..core.security import hash_password
from .models import Exercise, User
from .session import SessionLocal

BACKEND_ROOT = Path(__file__).resolve().parents[2]

_DEV_DEFAULT_ADMIN = ("admin@coachvision.test", "Admin1234")

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


def _seed_admin(db) -> None:
    """Create the admin account.

    In development the well-known dev credentials are used. Outside
    development the admin is only seeded when ADMIN_SEED_EMAIL and
    ADMIN_SEED_PASSWORD are explicitly set to non-default values, so the
    public default credentials never exist on a deployed instance.
    """
    email = settings.admin_seed_email.lower().strip()
    password = settings.admin_seed_password
    if settings.environment != "development" and (email, password) == _DEV_DEFAULT_ADMIN:
        return

    existing_admin = db.scalar(select(User).where(User.email == email))
    if not existing_admin:
        db.add(
            User(
                email=email,
                password_hash=hash_password(password),
                display_name="Admin User",
                role="admin",
            )
        )
    elif existing_admin.role != "admin":
        existing_admin.role = "admin"


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

        _seed_admin(db)
        db.commit()
    finally:
        db.close()

