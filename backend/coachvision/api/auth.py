"""Authentication endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .schemas import RefreshRequest, RegisterRequest, TokenPair
from ..core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from ..db.models import User
from ..db.session import get_db

router = APIRouter(prefix="/auth")


@router.post("/register", response_model=TokenPair)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> TokenPair:
    existing = db.scalar(select(User).where(User.email == payload.email.lower().strip()))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        email=payload.email.lower().strip(),
        password_hash=hash_password(payload.password),
        display_name=payload.display_name.strip(),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return TokenPair(
        access_token=create_access_token(str(user.id), {"role": user.role}),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.post("/login", response_model=TokenPair)
async def login(request: Request, db: Session = Depends(get_db)) -> TokenPair:
    content_type = request.headers.get("content-type", "").lower()
    email = ""
    password = ""

    if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        form_data = await request.form()
        email = str(form_data.get("username") or form_data.get("email") or "").lower().strip()
        password = str(form_data.get("password") or "")
    else:
        payload = await request.json()
        email = str(payload.get("email") or payload.get("username") or "").lower().strip()
        password = str(payload.get("password") or "")

    if not email or not password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Email/username and password are required",
        )

    user = db.scalar(select(User).where(User.email == email))
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    return TokenPair(
        access_token=create_access_token(str(user.id), {"role": user.role}),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.post("/refresh", response_model=TokenPair)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)) -> TokenPair:
    try:
        token_data = decode_token(payload.refresh_token)
        if token_data.get("token_type") != "refresh":
            raise ValueError("Invalid token type")
        user_id = token_data["sub"]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token") from exc

    user = db.get(User, UUID(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return TokenPair(
        access_token=create_access_token(user_id, {"role": user.role}),
        refresh_token=create_refresh_token(user_id),
    )


@router.post("/logout")
def logout() -> dict[str, str]:
    return {"status": "ok"}

