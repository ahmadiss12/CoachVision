"""End-to-end: a live WebSocket workout writes a benchmark clip.

Covers the wiring the unit tests cannot: that the recorder is actually reached
from the live pose path, that it stays silent when the feature is off, and that
an abandoned session still saves its frames.

Uses in-memory SQLite, following the pattern in test_account_deletion.py.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from coachvision.benchmark.clips import load_clip
from coachvision.core.config import settings
from coachvision.db.base import Base
from coachvision.db.models import Exercise
from coachvision.db.session import get_db
from coachvision.main import app
from coachvision.services.ws_persistence import normalize_exercise_id

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

LANDMARK_COUNT = 33
PASSWORD = "secret123"


def _override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def _pose(y: float) -> list[dict]:
    return [{"x": 0.5, "y": y, "presence": 0.9} for _ in range(LANDMARK_COUNT)]


class WsClipRecordingTest(unittest.TestCase):
    _previous_override = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._previous_override = app.dependency_overrides.get(get_db)
        app.dependency_overrides[get_db] = _override_get_db
        Base.metadata.create_all(bind=engine)

        # The live 'start' handler resolves the exercise against this table.
        db = TestingSessionLocal()
        db.add(Exercise(id=normalize_exercise_id("squat"), name="Squat"))
        db.commit()
        db.close()

        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._previous_override is None:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = cls._previous_override

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.fixtures = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _token(self, email: str) -> str:
        resp = self.client.post(
            "/v1/auth/register",
            json={
                "email": email,
                "password": PASSWORD,
                "display_name": "recorder",
                "role": "client",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["access_token"]

    def _recording_enabled(self, enabled: bool = True):
        return mock.patch.multiple(
            settings,
            clip_recording_enabled=enabled,
            clip_recording_dir=str(self.fixtures),
        )

    def _run_workout(self, token: str, frames: int = 6, send_end: bool = True) -> None:
        # Session persistence is stubbed: both of these compare a tz-aware `now`
        # against a stored datetime, and SQLite hands back naive datetimes where
        # Postgres returns aware ones. Those paths have their own tests; the
        # subject here is whether the recorder is reached and flushed.
        with (
            mock.patch("coachvision.api.ws_live.finalize_completed_workout"),
            mock.patch("coachvision.api.ws_live.abort_active_workout"),
        ):
            self._drive_socket(token, frames, send_end)

    def _drive_socket(self, token: str, frames: int, send_end: bool) -> None:
        with self.client.websocket_connect(f"/v1/ws/live?token={token}") as ws:
            ws.send_json({"type": "start", "exerciseName": "squat", "difficulty": "intermediate"})
            started = ws.receive_json()
            self.assertEqual(started["type"], "started")
            session_id = started["sessionId"]

            for i in range(frames):
                ws.send_json(
                    {
                        "type": "pose",
                        "sessionId": session_id,
                        "landmarks": _pose(0.5 + i * 0.01),
                        "timestampMs": 1_700_000_000_000 + i * 67,
                    }
                )
                ws.receive_json()

            if send_end:
                ws.send_json({"type": "end", "sessionId": session_id})
                self.assertEqual(ws.receive_json()["type"], "ended")

    def _written_clips(self, expected: int = 1, timeout: float = 5.0) -> list[Path]:
        """Wait for the recorder's flush, which deliberately runs off the loop.

        The clip is written in a worker thread once the session ends or the
        socket drops, so it lands slightly after the client is done -- that is
        the point, since the live loop must not block on disk. Polling here
        rather than sleeping keeps the test fast and non-flaky.
        """
        deadline = time.monotonic() + timeout
        while True:
            found = sorted(self.fixtures.glob("**/*.jsonl"))
            if len(found) >= expected or time.monotonic() > deadline:
                return found
            time.sleep(0.02)

    def test_live_workout_writes_a_loadable_clip(self) -> None:
        token = self._token("recorder-on@test.dev")
        with self._recording_enabled():
            self._run_workout(token, frames=6)

        clips = self._written_clips()
        self.assertEqual(len(clips), 1)

        clip = load_clip(clips[0])
        self.assertEqual(clip.meta.exercise, "squat")
        self.assertEqual(len(clip.frames), 6)
        self.assertEqual(len(clip.frames[0].landmarks), LANDMARK_COUNT)
        self.assertAlmostEqual(clip.frames[0].t, 0.0)
        self.assertGreater(clip.frames[-1].t, 0.0)

    def test_recorded_clip_is_unlabelled(self) -> None:
        token = self._token("recorder-label@test.dev")
        with self._recording_enabled():
            self._run_workout(token, frames=4)

        meta = json.loads(self._written_clips()[0].with_suffix(".meta.json").read_text())
        self.assertTrue(meta["needs_label"])
        self.assertIsNone(meta.get("true_reps"))

    def test_nothing_is_written_when_recording_is_off(self) -> None:
        token = self._token("recorder-off@test.dev")
        with self._recording_enabled(enabled=False):
            self._run_workout(token, frames=6)

        # Nothing to wait for, but give a flush the same window the positive
        # tests get, so this cannot pass merely by checking too early.
        self.assertEqual(self._written_clips(expected=1, timeout=1.0), [])

    def test_abandoned_session_still_saves_its_frames(self) -> None:
        """Disconnecting without 'end' is normal; those reps still happened."""
        token = self._token("recorder-drop@test.dev")
        with self._recording_enabled():
            self._run_workout(token, frames=5, send_end=False)

        clips = self._written_clips()
        self.assertEqual(len(clips), 1)
        self.assertEqual(len(load_clip(clips[0]).frames), 5)


if __name__ == "__main__":
    unittest.main()
