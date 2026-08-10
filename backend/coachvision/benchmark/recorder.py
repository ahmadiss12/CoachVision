"""Capture live pose frames into benchmark clips.

A benchmark clip is just the landmark stream the mobile app already sends over
the WebSocket, so recording one is a matter of saving what arrives. This module
is the dev-only path that does that.

Two properties are deliberate:

**It is off unless someone turns it on in development.** Pose landmarks are
still body data recorded from a person, so this is a data-collection feature and
is treated as one: disabled by default, and ``Settings`` refuses to start with
it enabled outside a development environment. Turning it on is a decision, never
a default.

**It does not write the app's own rep count into the label.** The recorder knows
what the counter said, and writing that into ``true_reps`` would be labelling
the data with the very thing the benchmark exists to measure -- every clip would
score 100% and the number would be meaningless. Clips land with
``needs_label: true`` and a human fills in what actually happened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..realtime.pipeline import IMPORTANT_LANDMARKS, _coerce_client_pose_landmarks
from .clips import ClipMeta, Frame, write_clip

logger = logging.getLogger(__name__)

MAX_FRAMES = 18_000
"""~20 minutes at 15 Hz. A stuck session shouldn't grow the buffer without end."""


@dataclass
class ClipRecorder:
    """Buffers one session's landmark frames, then writes them as a clip.

    Frames are held in memory and written once at the end: a disk write per
    frame would add I/O latency to the live loop, which is the one thing this
    must not do.
    """

    session_id: str
    exercise: str
    difficulty: str = "intermediate"
    fixtures_dir: Path = field(default=Path("fixtures"))

    _frames: list[Frame] = field(default_factory=list, init=False)
    _first_timestamp_ms: int | None = field(default=None, init=False)
    _dropped: int = field(default=0, init=False)
    _capped_warned: bool = field(default=False, init=False)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def record(self, raw_landmarks: object, timestamp_ms: int | None) -> None:
        """Buffer one frame. Never raises -- recording must not break a workout."""
        try:
            if len(self._frames) >= MAX_FRAMES:
                if not self._capped_warned:
                    logger.warning(
                        "Clip recording for session %s hit the %d frame cap; "
                        "further frames are not recorded.",
                        self.session_id,
                        MAX_FRAMES,
                    )
                    self._capped_warned = True
                return

            # Same coercion the pipeline runs, so the clip holds exactly what
            # the counter was fed -- dicts and arrays both normalize here.
            pose_lms = _coerce_client_pose_landmarks(raw_landmarks)
            if pose_lms is None:
                self._dropped += 1
                return

            self._frames.append(
                Frame(
                    t=self._relative_time(timestamp_ms),
                    landmarks=[[lm.x, lm.y, lm.presence] for lm in pose_lms],
                    confidence=self._mean_presence(pose_lms),
                )
            )
        except Exception:  # noqa: BLE001 - a recorder bug must not end a workout
            self._dropped += 1
            logger.exception("Clip recording failed for session %s", self.session_id)

    def _relative_time(self, timestamp_ms: int | None) -> float:
        """Seconds since the first recorded frame.

        Clip time is relative so a clip can be replayed on its own timeline. When
        the client sends no timestamp we fall back to the nominal 15 Hz live
        rate, which keeps the clip usable rather than collapsing every frame
        onto t=0.
        """
        if timestamp_ms is None:
            return len(self._frames) / 15.0
        if self._first_timestamp_ms is None:
            self._first_timestamp_ms = timestamp_ms
        return max(0.0, (timestamp_ms - self._first_timestamp_ms) / 1000.0)

    @staticmethod
    def _mean_presence(pose_lms: list) -> float:
        """The same scalar confidence the pipeline hands the counter."""
        values = [
            pose_lms[idx].presence
            for idx in IMPORTANT_LANDMARKS.values()
            if idx < len(pose_lms)
        ]
        return sum(values) / len(values) if values else 0.0

    def finish(self) -> Path | None:
        """Write the clip. Returns the path, or None if there was nothing to save.

        Safe to call twice (the disconnect path and the ``end`` path can race);
        the second call finds an empty buffer and does nothing.
        """
        if not self._frames:
            return None

        frames, self._frames = self._frames, []
        clip_id = f"rec_{self.exercise}_{self.session_id[:8]}"

        try:
            path = write_clip(
                self.fixtures_dir,
                clip_id,
                ClipMeta(
                    exercise=self.exercise,
                    level=self.difficulty,
                    camera_angle="unknown",
                    source="recorded",
                    needs_label=True,
                    notes=(
                        "Recorded live. Set true_reps (or true_hold_sec) to what "
                        "actually happened in this clip, set camera_angle, then "
                        "remove needs_label. Count from the recording itself, not "
                        "from what the app displayed."
                    ),
                ),
                frames,
            )
        except OSError:
            logger.exception("Could not write clip for session %s", self.session_id)
            return None

        logger.info(
            "Recorded clip %s (%d frames, %.1fs, %d dropped) -> %s",
            clip_id,
            len(frames),
            frames[-1].t - frames[0].t,
            self._dropped,
            path,
        )
        return path
