"""Authentication helpers (JWT/password)."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt
from passlib.context import CryptContext

from .config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(raw_password: str) -> str:
    return pwd_context.hash(raw_password)


def verify_password(raw_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(raw_password, hashed_password)


def create_access_token(subject: str, extra_claims: Dict[str, Any] | None = None) -> str:
    payload: Dict[str, Any] = {"sub": subject}
    if extra_claims:
        payload.update(extra_claims)
    expire_at = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_exp_minutes)
    payload["exp"] = expire_at
    payload["token_type"] = "access"
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(subject: str) -> str:
    payload: Dict[str, Any] = {"sub": subject, "token_type": "refresh"}
    expire_at = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_exp_days)
    payload["exp"] = expire_at
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])

