"""Live session registry: active workouts and their observer connections.

The client's phone owns the primary /ws/live connection. Trainers (and
admins) can attach as read-only observers via /ws/observe/{session_id};
every metrics frame sent to the client is mirrored to observers here.

State is in-process (single uvicorn worker), which matches how the app is
deployed today.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from fastapi import WebSocket


@dataclass
class LiveSessionInfo:
    session_id: str
    user_id: str
    exercise_id: str
    started_at: float = field(default_factory=time.time)


class LiveSessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, LiveSessionInfo] = {}
        self._observers: dict[str, list[WebSocket]] = {}

    # --- session lifecycle (called by the client's /ws/live handler) ---

    def register_session(self, session_id: str, user_id: str, exercise_id: str) -> None:
        self._sessions[session_id] = LiveSessionInfo(
            session_id=session_id, user_id=str(user_id), exercise_id=exercise_id
        )

    def unregister_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._observers.pop(session_id, None)

    def get_session(self, session_id: str) -> LiveSessionInfo | None:
        return self._sessions.get(session_id)

    def sessions_for_users(self, user_ids: set[str]) -> list[LiveSessionInfo]:
        return [info for info in self._sessions.values() if info.user_id in user_ids]

    # --- observers (trainer dashboard connections) ---

    def add_observer(self, session_id: str, websocket: WebSocket) -> None:
        self._observers.setdefault(session_id, []).append(websocket)

    def remove_observer(self, session_id: str, websocket: WebSocket) -> None:
        connections = self._observers.get(session_id)
        if not connections:
            return
        if websocket in connections:
            connections.remove(websocket)
        if not connections:
            self._observers.pop(session_id, None)

    def observer_count(self, session_id: str) -> int:
        return len(self._observers.get(session_id, []))

    async def broadcast(self, session_id: str, payload: dict) -> None:
        """Mirror a payload to every observer; drop connections that fail."""
        connections = self._observers.get(session_id)
        if not connections:
            return
        dead: list[WebSocket] = []
        for connection in list(connections):
            try:
                await connection.send_json(payload)
            except Exception:  # noqa: BLE001 - any send failure means the socket is gone
                dead.append(connection)
        for connection in dead:
            self.remove_observer(session_id, connection)


live_registry = LiveSessionRegistry()


class ChatRegistry:
    """Open chat sockets per user, for instant message delivery.

    Messages are persisted via REST before delivery, so a user with no open
    socket simply reads them from history next time — the socket is purely
    a live-push optimization, exactly like a SignalR hub group per user.
    """

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    def add(self, user_id: str, websocket: WebSocket) -> None:
        self._connections.setdefault(str(user_id), []).append(websocket)

    def remove(self, user_id: str, websocket: WebSocket) -> None:
        connections = self._connections.get(str(user_id))
        if not connections:
            return
        if websocket in connections:
            connections.remove(websocket)
        if not connections:
            self._connections.pop(str(user_id), None)

    def connection_count(self, user_id: str) -> int:
        return len(self._connections.get(str(user_id), []))

    async def send_to_user(self, user_id: str, payload: dict) -> None:
        connections = self._connections.get(str(user_id))
        if not connections:
            return
        dead: list[WebSocket] = []
        for connection in list(connections):
            try:
                await connection.send_json(payload)
            except Exception:  # noqa: BLE001
                dead.append(connection)
        for connection in dead:
            self.remove(user_id, connection)


chat_registry = ChatRegistry()
