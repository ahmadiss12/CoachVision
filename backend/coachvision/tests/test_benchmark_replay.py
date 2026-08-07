"""Tests for the counter accuracy benchmark harness."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from coachvision.benchmark import (
    ClipMeta,
    ReplayClock,
    aggregate,
    load_clip,
    replay_clip,
    score_clip,
)
from coachvision.benchmark.clips import Clip, Frame, write_clip
from coachvision.benchmark.report import ClipScore
from coachvision.benchmark.synthetic import plank_clip, squat_clip


def _clip(exercise: str, frames: list[Frame], **meta_kwargs) -> Clip:
    return Clip(
        clip_id=f"test_{exercise}",
        meta=ClipMeta(exercise=exercise, **meta_kwargs),
        frames=frames,
    )


class ReplayClockTest(unittest.TestCase):
    def test_reports_the_time_it_was_advanced_to(self) -> None:
        clock = ReplayClock(10.0)
        self.assertEqual(clock(), 10.0)
        clock.advance_to(12.5)
        self.assertEqual(clock(), 12.5)

    def test_rejects_going_backwards(self) -> None:
        """Out-of-order frames would silently produce negative durations."""
        clock = ReplayClock(5.0)
        with self.assertRaises(ValueError):
            clock.advance_to(4.0)


class ReplayIsClockIndependentTest(unittest.TestCase):
    """The property the whole clock refactor exists to provide."""

    def test_hold_time_is_correct_with_the_wall_clock_frozen(self) -> None:
        # If any counter reaches for time.time() again, this is what catches it:
        # with the wall clock stopped, a wall-clock-dependent counter reports 0.
        clip = _clip("plank", plank_clip(hold_sec=30.0, seed=1), true_hold_sec=30.0)

        with mock.patch("time.time", return_value=1_700_000_000.0):
            result = replay_clip(clip)

        self.assertAlmostEqual(result.counted_hold_sec, 30.0, delta=2.0)

    def test_replay_is_deterministic(self) -> None:
        clip = _clip("squat", squat_clip(reps=6, seed=2), true_reps=6)
        first = replay_clip(clip)
        second = replay_clip(clip)
        self.assertEqual(first.counted_reps, second.counted_reps)
        self.assertEqual(first.count_frames, second.count_frames)

    def test_counts_are_unaffected_by_clip_frame_rate(self) -> None:
        """Same movement, different capture rate, same rep count."""
        slow = _clip("squat", squat_clip(reps=5, fps=10.0, seed=3), true_reps=5)
        fast = _clip("squat", squat_clip(reps=5, fps=30.0, seed=3), true_reps=5)
        self.assertEqual(replay_clip(slow).counted_reps, replay_clip(fast).counted_reps)


class ReplayCountingTest(unittest.TestCase):
    def test_counts_full_depth_squats(self) -> None:
        clip = _clip("squat", squat_clip(reps=7, seed=4), true_reps=7)
        self.assertEqual(replay_clip(clip).counted_reps, 7)

    def test_rejects_partial_depth_squats(self) -> None:
        """Knee stops at 115 deg, above the intermediate 90 deg gate."""
        clip = _clip("squat", squat_clip(reps=6, bottom_angle=115.0, seed=5), true_reps=0)
        self.assertEqual(replay_clip(clip).counted_reps, 0)

    def test_records_the_frame_each_rep_was_counted_on(self) -> None:
        clip = _clip("squat", squat_clip(reps=4, seed=6), true_reps=4)
        result = replay_clip(clip)
        self.assertEqual(len(result.count_frames), result.counted_reps)
        self.assertEqual(result.count_frames, sorted(result.count_frames))

    def test_skips_frames_with_absent_landmarks(self) -> None:
        frames = squat_clip(reps=3, seed=7)
        # Blank out presence on a stretch of frames, as an occlusion would.
        for frame in frames[10:20]:
            for landmark in frame.landmarks:
                landmark[2] = 0.0
        result = replay_clip(_clip("squat", frames, true_reps=3))
        self.assertEqual(result.frames_skipped, 10)
        self.assertEqual(result.frames_seen, len(frames) - 10)


class ClipFormatTest(unittest.TestCase):
    def test_write_then_load_round_trips(self) -> None:
        frames = squat_clip(reps=2, seed=8)
        meta = ClipMeta(exercise="squat", true_reps=2, camera_angle="side")

        with tempfile.TemporaryDirectory() as tmp:
            path = write_clip(Path(tmp), "round_trip", meta, frames)
            loaded = load_clip(path)

        self.assertEqual(loaded.clip_id, "round_trip")
        self.assertEqual(loaded.meta.true_reps, 2)
        self.assertEqual(loaded.meta.camera_angle, "side")
        self.assertEqual(len(loaded.frames), len(frames))
        self.assertAlmostEqual(loaded.frames[-1].t, frames[-1].t, places=3)

    def test_metadata_must_carry_a_ground_truth(self) -> None:
        with self.assertRaises(ValueError):
            ClipMeta(exercise="squat")

    def test_missing_metadata_file_is_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orphan = Path(tmp) / "orphan.jsonl"
            orphan.write_text(json.dumps({"t": 0.0, "landmarks": []}) + "\n")
            with self.assertRaisesRegex(FileNotFoundError, "no ground truth"):
                load_clip(orphan)


class ScoringTest(unittest.TestCase):
    def _score(self, expected: float, counted: float, is_hold: bool = False) -> ClipScore:
        return ClipScore(
            clip_id="c",
            exercise="squat",
            is_hold=is_hold,
            expected=expected,
            counted=counted,
            camera_angle="side",
        )

    def test_reps_must_match_exactly(self) -> None:
        self.assertTrue(self._score(10, 10).exact)
        self.assertFalse(self._score(10, 11).exact)

    def test_holds_allow_two_seconds_of_slack(self) -> None:
        self.assertTrue(self._score(30, 31.5, is_hold=True).exact)
        self.assertFalse(self._score(30, 33.0, is_hold=True).exact)

    def test_error_sign_separates_over_from_under_counting(self) -> None:
        self.assertGreater(self._score(10, 12).error, 0)
        self.assertLess(self._score(10, 8).error, 0)

    def test_scores_a_hold_clip_against_its_hold_label(self) -> None:
        clip = _clip("plank", plank_clip(hold_sec=25.0, seed=9), true_hold_sec=25.0)
        score = score_clip(clip, replay_clip(clip))
        self.assertTrue(score.is_hold)
        self.assertEqual(score.expected, 25.0)


class AggregateTest(unittest.TestCase):
    def _scores(self, exercise: str, pairs: list[tuple[float, float]]) -> list[ClipScore]:
        return [
            ClipScore(
                clip_id=f"{exercise}_{i}",
                exercise=exercise,
                is_hold=False,
                expected=e,
                counted=c,
                camera_angle="side",
            )
            for i, (e, c) in enumerate(pairs)
        ]

    def test_rates_partition_the_clips(self) -> None:
        summary = aggregate(self._scores("squat", [(10, 10), (10, 12), (10, 8), (10, 10)]))[0]
        self.assertEqual(summary.clips, 4)
        self.assertAlmostEqual(summary.exact_rate, 0.5)
        self.assertAlmostEqual(summary.over_rate, 0.25)
        self.assertAlmostEqual(summary.under_rate, 0.25)
        self.assertAlmostEqual(
            summary.exact_rate + summary.over_rate + summary.under_rate, 1.0
        )

    def test_mean_absolute_error_does_not_let_errors_cancel(self) -> None:
        """+2 and -2 is 100% wrong, not an average of zero."""
        summary = aggregate(self._scores("squat", [(10, 12), (10, 8)]))[0]
        self.assertAlmostEqual(summary.mae, 2.0)
        self.assertEqual(summary.exact_rate, 0.0)

    def test_worst_exercise_is_reported_first(self) -> None:
        scores = self._scores("squat", [(10, 10), (10, 10)])
        scores += self._scores("lunge", [(10, 4), (10, 10)])
        self.assertEqual([s.exercise for s in aggregate(scores)], ["lunge", "squat"])

    def test_worst_clip_is_the_largest_absolute_error(self) -> None:
        summary = aggregate(self._scores("squat", [(10, 9), (10, 3), (10, 11)]))[0]
        self.assertEqual(summary.worst.counted, 3)


if __name__ == "__main__":
    unittest.main()
