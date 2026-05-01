"""WebSocket connection manager for live sessions."""

from collections import defaultdict

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._sessions[session_id].append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        if session_id not in self._sessions:
            return
        if websocket in self._sessions[session_id]:
            self._sessions[session_id].remove(websocket)
        if not self._sessions[session_id]:
            self._sessions.pop(session_id, None)

    async def broadcast(self, session_id: str, payload: dict) -> None:
        for connection in list(self._sessions.get(session_id, [])):
            await connection.send_json(payload)

