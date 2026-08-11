"""Hold-time accounting for plank and wall sit.

These counters used to accumulate hold time as a hard-coded ``1.0 / 30`` per
frame, which is only correct if the camera happens to run at exactly 30 fps.
The live app runs at roughly 15 Hz, so a plank was credited at half speed --
and a capture above 30 fps would have been credited at more than double.

Hold time now comes from the injected clock, so it reflects how long the person
actually held the position rather than how fast frames arrived.
"""

import random
import unittest

from coachvision.ai.counters.plank import PlankCounter
from coachvision.ai.counters.wall_sit import WallSitCounter
from coachvision.benchmark.clips import Clip, ClipMeta, Frame
from coachvision.benchmark.replay import replay_clip
from coachvision.benchmark.synthetic import PRESENCE, _plank_frame, plank_clip
from coachvision.realtime.pipeline import IMPORTANT_LANDMARKS


def _clip(frames: list[Frame], true_hold_sec: float) -> Clip:
    return Clip(
        clip_id="hold",
        meta=ClipMeta(exercise="plank", true_hold_sec=true_hold_sec),
        frames=frames,
    )


def _joints(landmarks: list[list[float]]) -> dict[str, tuple[float, float]]:
    return {
        name: (landmarks[idx][0], landmarks[idx][1])
        for name, idx in IMPORTANT_LANDMARKS.items()
    }


class FrameRateIndependenceTest(unittest.TestCase):
    def test_same_hold_scores_the_same_at_any_frame_rate(self) -> None:
        for fps in (10.0, 15.0, 30.0, 60.0):
            with self.subTest(fps=fps):
                clip = _clip(plank_clip(hold_sec=40.0, fps=fps, seed=1), 40.0)
                self.assertAlmostEqual(
                    replay_clip(clip).counted_hold_sec, 40.0, delta=1.5
                )

    def test_the_apps_own_rate_is_not_halved(self) -> None:
        """The regression itself: 15 Hz used to report half the real time."""
        clip = _clip(plank_clip(hold_sec=30.0, fps=15.0, seed=2), 30.0)
        self.assertAlmostEqual(replay_clip(clip).counted_hold_sec, 30.0, delta=1.5)

    def test_a_fast_capture_is_not_doubled(self) -> None:
        clip = _clip(plank_clip(hold_sec=30.0, fps=60.0, seed=3), 30.0)
        self.assertAlmostEqual(replay_clip(clip).counted_hold_sec, 30.0, delta=1.5)


class BrokenHoldTest(unittest.TestCase):
    """The case that made the bug visible: a plank whose form breaks."""

    def test_segments_sum_to_the_total_time_held(self) -> None:
        clip = _clip(plank_clip(hold_sec=60.0, breaks=2, fps=15.0, seed=4), 60.0)
        self.assertAlmostEqual(replay_clip(clip).counted_hold_sec, 60.0, delta=2.0)

    def test_breaks_do_not_count_toward_hold_time(self) -> None:
        """Time spent with hips sagging is not time spent planking."""
        frames = plank_clip(hold_sec=30.0, breaks=2, break_sec=10.0, fps=15.0, seed=5)
        counted = replay_clip(_clip(frames, 30.0)).counted_hold_sec

        self.assertAlmostEqual(counted, 30.0, delta=2.0)
        # The clip runs ~52s end to end; only the ~30s of good form counts.
        self.assertLess(counted, 40.0)


class InProgressHoldTest(unittest.TestCase):
    def test_a_session_ended_mid_hold_still_reports_it(self) -> None:
        """Ending the workout while still holding must not discard the hold."""
        frames = plank_clip(hold_sec=25.0, fps=15.0, seed=6)
        # plank_clip ends with a second of sag; drop it so the clip stops while
        # the hold is still in progress, as it does when a user hits "end".
        frames = frames[: -int(1.0 * 15)]

        self.assertAlmostEqual(
            replay_clip(_clip(frames, 25.0)).counted_hold_sec, 25.0, delta=2.0
        )

    def test_live_total_rises_while_holding(self) -> None:
        """The on-screen SEC counter must keep moving during a hold."""
        now = [1000.0]
        counter = PlankCounter(clock=lambda: now[0])
        rng = random.Random(7)

        counter.update(_joints(_plank_frame(178.0, rng, 0.0005)), PRESENCE)

        readings = []
        for _ in range(5):
            now[0] += 2.0
            counter.update(_joints(_plank_frame(178.0, rng, 0.0005)), PRESENCE)
            readings.append(counter.get_total_hold_time())

        self.assertEqual(readings, sorted(readings), "counter went backwards")
        self.assertAlmostEqual(readings[-1] - readings[0], 8.0, delta=0.5)


class HoldAccumulationTest(unittest.TestCase):
    """Both hold counters carried the identical hard-coded rate."""

    def test_plank_accumulates_elapsed_time(self) -> None:
        now = [100.0]
        counter = PlankCounter(clock=lambda: now[0])
        counter._hold_start_time = 100.0
        counter._end_hold_session(130.0)
        self.assertAlmostEqual(counter.get_total_hold_time(), 30.0)

    def test_wall_sit_accumulates_elapsed_time(self) -> None:
        now = [100.0]
        counter = WallSitCounter(clock=lambda: now[0])
        counter._hold_start_time = 100.0
        counter._end_hold_session(130.0)
        self.assertAlmostEqual(counter.get_total_hold_time(), 30.0)

    def test_consecutive_holds_add_up(self) -> None:
        counter = PlankCounter(clock=lambda: 0.0)
        counter._hold_start_time = 0.0
        counter._end_hold_session(20.0)
        counter._hold_start_time = 30.0
        counter._end_hold_session(45.0)
        self.assertAlmostEqual(counter.get_total_hold_time(), 35.0)

    def test_short_holds_still_count_toward_total_time(self) -> None:
        """A wobble below min_hold_time is not a session, but it was still held."""
        counter = PlankCounter(clock=lambda: 0.0)
        counter._hold_start_time = 0.0
        counter._end_hold_session(2.0)  # under the 10s intermediate minimum

        self.assertEqual(counter.hold_count, 0)
        self.assertAlmostEqual(counter.get_total_hold_time(), 2.0)


if __name__ == "__main__":
    unittest.main()
