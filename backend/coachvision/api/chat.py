"""Trainer-client chat: REST history/sending + WebSocket live delivery.

Design (deliberately SignalR-like, on plain WebSockets):
- Sending goes through REST so every message is persisted before delivery.
- Each user may keep a /ws/chat socket open; new messages are pushed to the
  recipient's sockets (and the sender's other devices) instantly.
- Conversations exist only along active trainer_clients links — the same
  isolation rule as every other trainer/client feature.
"""

from datetime import datetime, timezone
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session as DbSession

from .deps import get_current_user
from .schemas import ChatMessageResponse, ChatSendRequest, ConversationResponse
from ..core.security import decode_token
from ..db.models import ChatMessage, TrainerClient, User
from ..db.session import get_db
from ..realtime.connection_manager import chat_registry

router = APIRouter()


def _has_active_link(db: DbSession, user_a: UUID, user_b: UUID) -> bool:
    return (
        db.scalar(
            select(TrainerClient.id).where(
                or_(
                    (TrainerClient.trainer_id == user_a) & (TrainerClient.client_id == user_b),
                    (TrainerClient.trainer_id == user_b) & (TrainerClient.client_id == user_a),
                ),
                TrainerClient.status == "active",
            )
        )
        is not None
    )


def _partners_for(db: DbSession, user: User) -> list[User]:
    """Everyone this user can chat with: active links in either direction."""
    partner_ids = set(
        db.scalars(
            select(TrainerClient.client_id).where(
                TrainerClient.trainer_id == user.id, TrainerClient.status == "active"
            )
        ).all()
    ) | set(
        db.scalars(
            select(TrainerClient.trainer_id).where(
                TrainerClient.client_id == user.id, TrainerClient.status == "active"
            )
        ).all()
    )
    if not partner_ids:
        return []
    return list(db.scalars(select(User).where(User.id.in_(partner_ids))).all())


@router.get("/chat/conversations", response_model=list[ConversationResponse])
def list_conversations(
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ConversationResponse]:
    partners = _partners_for(db, current_user)
    out: list[ConversationResponse] = []
    for partner in partners:
        last = db.scalars(
            select(ChatMessage)
            .where(
                or_(
                    (ChatMessage.sender_id == current_user.id)
                    & (ChatMessage.recipient_id == partner.id),
                    (ChatMessage.sender_id == partner.id)
                    & (ChatMessage.recipient_id == current_user.id),
                )
            )
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        ).first()
        unread = db.scalar(
            select(func.count(ChatMessage.id)).where(
                ChatMessage.sender_id == partner.id,
                ChatMessage.recipient_id == current_user.id,
                ChatMessage.read_at.is_(None),
            )
        )
        out.append(
            ConversationResponse(
                partner_id=partner.id,
                partner_name=partner.display_name,
                partner_role=partner.role,
                last_message=last.body if last else None,
                last_at=last.created_at if last else None,
                unread_count=int(unread or 0),
            )
        )
    out.sort(key=lambda c: (c.last_at is not None, c.last_at), reverse=True)
    return out


@router.get("/chat/unread-count")
def unread_count(
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, int]:
    total = db.scalar(
        select(func.count(ChatMessage.id)).where(
            ChatMessage.recipient_id == current_user.id,
            ChatMessage.read_at.is_(None),
        )
    )
    return {"unread": int(total or 0)}


@router.get("/chat/{partner_id}/messages", response_model=list[ChatMessageResponse])
def get_messages(
    partner_id: UUID,
    limit: int = 50,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ChatMessageResponse]:
    if not _has_active_link(db, current_user.id, partner_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    rows = db.scalars(
        select(ChatMessage)
        .where(
            or_(
                (ChatMessage.sender_id == current_user.id)
                & (ChatMessage.recipient_id == partner_id),
                (ChatMessage.sender_id == partner_id)
                & (ChatMessage.recipient_id == current_user.id),
            )
        )
        .order_by(ChatMessage.created_at.desc())
        .limit(max(1, min(limit, 200)))
    ).all()

    # Opening the conversation marks the partner's messages as read.
    now = datetime.now(timezone.utc)
    for row in rows:
        if row.recipient_id == current_user.id and row.read_at is None:
            row.read_at = now
    db.commit()

    return [ChatMessageResponse.model_validate(row) for row in reversed(rows)]


@router.post(
    "/chat/{partner_id}/messages",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def send_message(
    partner_id: UUID,
    payload: ChatSendRequest,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatMessageResponse:
    if not _has_active_link(db, current_user.id, partner_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    message = ChatMessage(
        sender_id=current_user.id,
        recipient_id=partner_id,
        body=payload.body.strip(),
    )
    db.add(message)
    db.commit()
    db.refresh(message)

    response = ChatMessageResponse.model_validate(message)
    push = {
        "type": "message",
        **response.model_dump(by_alias=True, mode="json"),
        "senderName": current_user.display_name,
    }
    # Instant delivery to the partner, and to the sender's other devices.
    await chat_registry.send_to_user(str(partner_id), push)
    await chat_registry.send_to_user(str(current_user.id), push)
    return response


@router.websocket("/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    token: str = Query(..., description="JWT access token"),
    db: DbSession = Depends(get_db),
) -> None:
    await websocket.accept()
    try:
        payload = decode_token(token)
        if payload.get("token_type") != "access":
            raise ValueError("Invalid token type")
        user_id = UUID(str(payload["sub"]))
    except Exception:  # noqa: BLE001
        await websocket.close(code=4401)
        return

    if db.get(User, user_id) is None:
        await websocket.close(code=4401)
        return

    key = str(user_id)
    chat_registry.add(key, websocket)
    await websocket.send_json({"type": "connected"})
    try:
        while True:
            # Receive-only channel; inbound frames are keep-alives.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        chat_registry.remove(key, websocket)
