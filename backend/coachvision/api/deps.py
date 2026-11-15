"""Shared API dependencies."""

from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ..core.entitlements import Entitlements, minimum_plan_for, resolve_entitlements
from ..core.security import decode_token
from ..db.models import User
from ..db.session import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    try:
        payload = decode_token(token)
        if payload.get("token_type") != "access":
            raise ValueError("Invalid token type")
        user_id = payload.get("sub")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    user = db.get(User, UUID(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_role(*roles: str):
    """Dependency factory: only allow users whose role is in ``roles``.

    The role is read from the database row (via get_current_user), not the JWT
    claim, so a role change takes effect immediately without reissuing tokens.
    """

    def dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {', '.join(roles)}",
            )
        return current_user

    return dependency



def get_entitlements(current_user: User = Depends(get_current_user)) -> Entitlements:
    """What the caller's subscription plan allows right now."""
    return resolve_entitlements(current_user.subscription_plan, current_user.role)


def require_feature(feature: str):
    """Dependency factory: only allow callers whose plan includes ``feature``.

    Responds 402 so clients can tell "upgrade needed" apart from the 403 that
    ``require_role`` returns.
    """

    def dependency(entitlements: Entitlements = Depends(get_entitlements)) -> Entitlements:
        if not entitlements.has_feature(feature):
            needed = minimum_plan_for(feature) or "a higher"
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"This feature requires the {needed} plan",
            )
        return entitlements

    return dependency
