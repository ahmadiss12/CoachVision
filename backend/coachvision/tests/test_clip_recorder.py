"""Tests for live clip recording (the benchmark dataset capture path)."""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from coachvision.benchmark.clips import iter_clips, load_clip
from coachvision.benchmark.recorder import MAX_FRAMES, ClipRecorder
from coachvision.core.config import Settings

LANDMARK_COUNT = 33


def _dict_landmarks(y: float = 0.5, presence: float = 0.9) -> list[dict]:
    return [{"x": 0.5, "y": y, "presence": presence} for _ in range(LANDMARK_COUNT)]


def _array_landmarks(y: float = 0.5, presence: float = 0.9) -> list[list[float]]:
    return [[0.5, y, presence] for _ in range(LANDMARK_COUNT)]


class RecorderCaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.fixtures = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _recorder(self, exercise: str = "squat") -> ClipRecorder:
        return ClipRecorder(
            session_id="abcdef12-3456-7890-abcd-ef1234567890",
            exercise=exercise,
            fixtures_dir=self.fixtures,
        )

    def test_accepts_both_landmark_wire_formats(self) -> None:
        """Clients may send dicts or arrays; both must record identically."""
        as_dicts = self._recorder()
        as_arrays = self._recorder()
        as_dicts.record(_dict_landmarks(), 0)
        as_arrays.record(_array_landmarks(), 0)
        self.assertEqual(as_dicts.frame_count, 1)
        self.assertEqual(as_arrays.frame_count, 1)

    def test_clip_time_is_relative_to_the_first_frame(self) -> None:
        recorder = self._recorder()
        for ts in (1_700_000_000_000, 1_700_000_000_100, 1_700_000_000_250):
            recorder.record(_dict_landmarks(), ts)

        path = recorder.finish()
        clip = load_clip(path)
        self.assertAlmostEqual(clip.frames[0].t, 0.0)
        self.assertAlmostEqual(clip.frames[1].t, 0.1, places=3)
        self.assertAlmostEqual(clip.frames[2].t, 0.25, places=3)

    def test_missing_timestamps_fall_back_to_the_nominal_rate(self) -> None:
        """Without this every frame lands on t=0 and the clip is unusable."""
        recorder = self._recorder()
        for _ in range(3):
            recorder.record(_dict_landmarks(), None)

        clip = load_clip(recorder.finish())
        self.assertAlmostEqual(clip.frames[1].t, 1 / 15, places=3)
        self.assertGreater(clip.frames[2].t, clip.frames[1].t)

    def test_confidence_is_the_mean_presence_of_tracked_joints(self) -> None:
        recorder = self._recorder()
        recorder.record(_dict_landmarks(presence=0.8), 0)
        clip = load_clip(recorder.finish())
        self.assertAlmostEqual(clip.frames[0].confidence, 0.8, places=2)

    def test_malformed_landmarks_are_dropped_not_raised(self) -> None:
        """A recorder fault must never interrupt someone's workout."""
        recorder = self._recorder()
        recorder.record("not landmarks", 0)
        recorder.record([{"x": 0.1}], 0)  # missing y
        recorder.record(_dict_landmarks()[:5], 0)  # too few landmarks
        recorder.record(None, 0)
        self.assertEqual(recorder.frame_count, 0)

        recorder.record(_dict_landmarks(), 0)
        self.assertEqual(recorder.frame_count, 1)

    def test_frame_buffer_is_capped(self) -> None:
        recorder = self._recorder()
        for _ in range(MAX_FRAMES + 50):
            recorder.record(_array_landmarks(), None)
        self.assertEqual(recorder.frame_count, MAX_FRAMES)


class RecorderOutputTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.fixtures = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _recorded(self, exercise: str = "squat", frames: int = 5) -> Path:
        recorder = ClipRecorder(
            session_id="abcdef12-3456-7890-abcd-ef1234567890",
            exercise=exercise,
            fixtures_dir=self.fixtures,
        )
        for i in range(frames):
            recorder.record(_dict_landmarks(y=0.5 + i * 0.01), i * 67)
        return recorder.finish()

    def test_writes_a_clip_the_benchmark_can_load(self) -> None:
        clip = load_clip(self._recorded())
        self.assertEqual(clip.meta.exercise, "squat")
        self.assertEqual(len(clip.frames), 5)
        self.assertEqual(len(clip.frames[0].landmarks), LANDMARK_COUNT)

    def test_recorded_clip_needs_a_human_label(self) -> None:
        meta = json.loads(self._recorded().with_suffix(".meta.json").read_text())
        self.assertTrue(meta["needs_label"])

    def test_does_not_write_the_apps_own_count_as_ground_truth(self) -> None:
        """Labelling with the counter's output would make every clip score 100%."""
        meta = json.loads(self._recorded().with_suffix(".meta.json").read_text())
        self.assertIsNone(meta.get("true_reps"))
        self.assertIsNone(meta.get("true_hold_sec"))

    def test_unlabelled_clips_are_skipped_and_reported(self) -> None:
        self._recorded()
        unlabelled: list[str] = []
        clips = list(iter_clips(self.fixtures, unlabelled=unlabelled))
        self.assertEqual(clips, [])
        self.assertEqual(len(unlabelled), 1)

    def test_labelling_a_recorded_clip_makes_it_benchmarkable(self) -> None:
        meta_path = self._recorded().with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        meta["true_reps"] = 3
        meta["needs_label"] = False
        meta_path.write_text(json.dumps(meta))

        clips = list(iter_clips(self.fixtures))
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].meta.true_reps, 3)

    def test_nothing_recorded_writes_nothing(self) -> None:
        recorder = ClipRecorder(
            session_id="s", exercise="squat", fixtures_dir=self.fixtures
        )
        self.assertIsNone(recorder.finish())
        self.assertEqual(list(self.fixtures.glob("**/*.jsonl")), [])

    def test_finishing_twice_is_harmless(self) -> None:
        """The end message and the disconnect handler can both fire."""
        recorder = ClipRecorder(
            session_id="s", exercise="squat", fixtures_dir=self.fixtures
        )
        recorder.record(_dict_landmarks(), 0)
        self.assertIsNotNone(recorder.finish())
        self.assertIsNone(recorder.finish())


class RecordingIsGatedTest(unittest.TestCase):
    """Pose landmarks are body data; recording them is opt-in and dev-only."""

    _SECRET = "x" * 64

    def test_off_by_default(self) -> None:
        self.assertFalse(Settings(jwt_secret_key=self._SECRET).clip_recording_enabled)

    def test_allowed_in_development(self) -> None:
        settings = Settings(environment="development", clip_recording_enabled=True)
        self.assertTrue(settings.clip_recording_enabled)

    def test_refused_in_production(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            Settings(
                environment="production",
                clip_recording_enabled=True,
                jwt_secret_key=self._SECRET,
            )
        self.assertIn("CLIP_RECORDING_ENABLED", str(ctx.exception))

    def test_refused_in_staging(self) -> None:
        with self.assertRaises(ValidationError):
            Settings(
                environment="staging",
                clip_recording_enabled=True,
                jwt_secret_key=self._SECRET,
            )


if __name__ == "__main__":
    unittest.main()
