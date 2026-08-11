"""Replay a recorded clip through the live counting path.

Two properties matter here, and both are easy to get wrong:

1. **Replay the same path production uses.** The live pipeline smooths every
   landmark with a One Euro filter *before* the counter sees it
   (``realtime/pipeline.py``). Benchmarking a bare counter would measure
   something no user ever experiences, so the constants and the filter class
   are imported from the pipeline rather than copied -- retune the pipeline and
   the benchmark follows automatically.

2. **Replay on clip time, not wall-clock time.** Every counter stamps reps and
   hold durations from an injected clock. Here that clock is driven by the
   recorded frame timestamps, so a 60-second clip scores identically whether it
   replays in 60 seconds or 300 milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ai.counters.dispatcher import ExerciseDispatcher
from ..ai.utils.geometry import LandmarkFilter
from ..realtime.pipeline import (
    IMPORTANT_LANDMARKS,
    LANDMARK_FILTER_BETA,
    LANDMARK_FILTER_MIN_CUTOFF,
)
from .clips import Clip

# MediaPipe reports presence per landmark; below this the joint is too
# uncertain to trust. Matches the counters' own default min_confidence.
MIN_PRESENCE = 0.5


class ReplayClock:
    """A clock the harness advances by hand.

    Counters call this instead of :func:`time.time`, so replay speed has no
    effect on the reps and hold durations they record.
    """

    __slots__ = ("_now",)

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance_to(self, t: float) -> None:
        """Jump the clock to ``t``. Never moves backwards."""
        if t < self._now:
            raise ValueError(f"replay clock cannot go backwards: {self._now} -> {t}")
        self._now = t


@dataclass
class ReplayResult:
    """What the counter produced for one clip."""

    clip_id: str
    exercise: str
    counted_reps: int = 0
    counted_hold_sec: float = 0.0
    frames_seen: int = 0
    frames_skipped: int = 0
    """Frames dropped before reaching the counter (missing or low-presence joints)."""
    count_frames: list[int] = field(default_factory=list)
    """Frame index at which each rep was counted -- for count-latency analysis."""
    angles: list[float] = field(default_factory=list, repr=False)
    """The counter's primary joint angle per processed frame.

    Raw geometry, not a decision: the labelling tool plots this so a human can
    count the reps in a recording themselves.
    """


def _joints_for_frame(
    landmarks: list[list[float]],
    t: float,
    filters: dict[str, LandmarkFilter],
) -> dict[str, tuple[float, float]] | None:
    """Convert MediaPipe landmarks to the ``{joint: (x, y)}`` counters expect.

    Mirrors ``LiveSessionState._extract_landmarks``, but takes the timestamp as
    an argument instead of reading the wall clock.
    """
    joints: dict[str, tuple[float, float]] = {}
    for name, idx in IMPORTANT_LANDMARKS.items():
        if idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        presence = lm[2] if len(lm) > 2 else 1.0
        if presence < MIN_PRESENCE:
            return None
        joints[name] = filters[name].filter(float(lm[0]), float(lm[1]), t)
    return joints


def replay_clip(clip: Clip, level: str | None = None) -> ReplayResult:
    """Run one clip through the counting path and report what it counted.

    Args:
        clip: The labelled recording to replay.
        level: Difficulty preset override. Defaults to the clip's own label, so
            a clip recorded at "beginner" is scored against beginner thresholds
            unless a sweep explicitly asks otherwise.
    """
    level = level or clip.meta.level
    clock = ReplayClock(clip.frames[0].t)

    dispatcher = ExerciseDispatcher()
    dispatcher.set_exercise(clip.meta.exercise, level=level, clock=clock)

    filters = {
        name: LandmarkFilter(
            min_cutoff=LANDMARK_FILTER_MIN_CUTOFF,
            beta=LANDMARK_FILTER_BETA,
        )
        for name in IMPORTANT_LANDMARKS
    }

    result = ReplayResult(clip_id=clip.clip_id, exercise=clip.meta.exercise)
    previous_count = 0

    for index, frame in enumerate(clip.frames):
        clock.advance_to(frame.t)

        joints = _joints_for_frame(frame.landmarks, frame.t, filters)
        if joints is None:
            result.frames_skipped += 1
            continue

        count, _state, angle = dispatcher.update(joints, frame.confidence)
        result.frames_seen += 1
        result.angles.append(float(angle))

        if count > previous_count:
            result.count_frames.extend([index] * (count - previous_count))
            previous_count = count

    if dispatcher.is_static_exercise():
        # Reuse the same summary the WebSocket persists at end-of-session, so
        # the benchmark scores the number the app actually stores.
        summary = dispatcher.export_session_data().get("summary", {})
        result.counted_hold_sec = float(summary.get("total_seconds") or 0.0)
    else:
        result.counted_reps = previous_count

    return result
