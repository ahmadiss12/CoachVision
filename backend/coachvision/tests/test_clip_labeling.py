"""Tests for filling in clip ground truth."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from coachvision.benchmark.clips import ClipMeta, load_clip, write_clip
from coachvision.benchmark.labeling import (
    angle_plot,
    is_hold_exercise,
    survey,
    unlabelled_clip_paths,
    write_label,
)
from coachvision.benchmark.synthetic import squat_clip


def _dips(reps: int, samples: int = 400) -> list[float]:
    """A joint angle trace with exactly ``reps`` flexion dips."""
    return [
        170.0 - 80.0 * (1.0 - math.cos(2.0 * math.pi * reps * i / samples)) / 2.0
        for i in range(samples)
    ]


class AnglePlotTest(unittest.TestCase):
    def test_dips_are_visible_in_the_plot(self) -> None:
        """The bottom row should show one run of marks per repetition."""
        plot = angle_plot(_dips(5), width=60, height=9)
        bottom = plot.splitlines()[-2]
        groups = [chunk for chunk in bottom.split("|")[-1].split(" ") if chunk]
        self.assertEqual(len(groups), 5)

    def test_rep_count_changes_the_number_of_dips(self) -> None:
        def dip_count(reps: int) -> int:
            bottom = angle_plot(_dips(reps), width=60, height=9).splitlines()[-2]
            return len([c for c in bottom.split("|")[-1].split(" ") if c])

        self.assertEqual(dip_count(3), 3)
        self.assertEqual(dip_count(7), 7)

    def test_plot_is_the_requested_size(self) -> None:
        plot = angle_plot(_dips(4), width=40, height=6)
        lines = plot.splitlines()
        self.assertEqual(len(lines), 7)  # height rows plus the axis
        self.assertTrue(all(len(line.split("|")[-1]) == 40 for line in lines[:6]))

    def test_flat_trace_says_there_is_nothing_to_count(self) -> None:
        self.assertIn("flat", angle_plot([90.0] * 50))

    def test_short_trace_does_not_crash(self) -> None:
        self.assertIn("not enough frames", angle_plot([]))
        self.assertIn("not enough frames", angle_plot([90.0]))

    def test_a_replayed_clip_plots_its_actual_reps(self) -> None:
        """End to end: six real squats produce six countable dips."""
        from coachvision.benchmark.clips import Clip
        from coachvision.benchmark.replay import replay_clip

        clip = Clip(
            clip_id="six",
            meta=ClipMeta(exercise="squat", true_reps=6),
            frames=squat_clip(reps=6, seed=11),
        )
        plot = angle_plot(replay_clip(clip).angles, width=60, height=9)
        bottom = plot.splitlines()[-2]
        groups = [chunk for chunk in bottom.split("|")[-1].split(" ") if chunk]
        self.assertEqual(len(groups), 6)


class WriteLabelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _unlabelled(self, exercise: str = "squat") -> Path:
        path = write_clip(
            self.dir,
            f"rec_{exercise}_test",
            ClipMeta(
                exercise=exercise,
                needs_label=True,
                source="recorded",
                notes="Recorded live. Set true_reps ...",
            ),
            squat_clip(reps=3, seed=1),
        )
        return path.with_suffix(".meta.json")

    def test_rep_label_clears_the_needs_label_flag(self) -> None:
        meta_path = self._unlabelled()
        write_label(meta_path, exercise="squat", value=9, camera_angle="side")

        meta = json.loads(meta_path.read_text())
        self.assertEqual(meta["true_reps"], 9)
        self.assertFalse(meta["needs_label"])
        self.assertEqual(meta["camera_angle"], "side")

    def test_hold_label_uses_seconds(self) -> None:
        meta_path = self._unlabelled("plank")
        write_label(meta_path, exercise="plank", value=42.5, camera_angle="side")

        meta = json.loads(meta_path.read_text())
        self.assertEqual(meta["true_hold_sec"], 42.5)
        self.assertNotIn("true_reps", meta)

    def test_reps_are_stored_as_a_whole_number(self) -> None:
        meta_path = self._unlabelled()
        write_label(meta_path, exercise="squat", value=7.0, camera_angle="side")
        self.assertIsInstance(json.loads(meta_path.read_text())["true_reps"], int)

    def test_recorder_instructions_are_cleared_once_labelled(self) -> None:
        """The recorder's 'go label this' note is stale after labelling."""
        meta_path = self._unlabelled()
        write_label(meta_path, exercise="squat", value=5, camera_angle="side")
        self.assertNotIn("notes", json.loads(meta_path.read_text()))

    def test_supplied_notes_are_kept(self) -> None:
        meta_path = self._unlabelled()
        write_label(
            meta_path,
            exercise="squat",
            value=5,
            camera_angle="front",
            notes="last rep was shallow",
        )
        self.assertEqual(
            json.loads(meta_path.read_text())["notes"], "last rep was shallow"
        )

    def test_labelled_clip_becomes_loadable_and_benchmarkable(self) -> None:
        meta_path = self._unlabelled()
        write_label(meta_path, exercise="squat", value=3, camera_angle="side")

        clip = load_clip(meta_path.with_suffix("").with_suffix(".jsonl"))
        self.assertFalse(clip.meta.needs_label)
        self.assertEqual(clip.meta.true_reps, 3)


class SurveyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _clip(self, exercise: str, clip_id: str, needs_label: bool) -> Path:
        return write_clip(
            self.dir,
            clip_id,
            ClipMeta(
                exercise=exercise,
                needs_label=needs_label,
                true_reps=None if needs_label else 5,
            ),
            squat_clip(reps=2, seed=2),
        )

    def test_counts_labelled_and_pending_per_exercise(self) -> None:
        self._clip("squat", "a", needs_label=False)
        self._clip("squat", "b", needs_label=True)
        self._clip("plank", "c", needs_label=True)

        rows = {row.exercise: row for row in survey(self.dir)}
        self.assertEqual((rows["squat"].labelled, rows["squat"].pending), (1, 1))
        self.assertEqual((rows["plank"].labelled, rows["plank"].pending), (0, 1))

    def test_reports_how_many_more_clips_are_needed(self) -> None:
        self._clip("squat", "a", needs_label=False)
        self.assertEqual(survey(self.dir)[0].still_needed, 14)

    def test_empty_directory_surveys_to_nothing(self) -> None:
        self.assertEqual(survey(self.dir), [])

    def test_only_unlabelled_clips_are_queued(self) -> None:
        self._clip("squat", "done", needs_label=False)
        self._clip("squat", "todo", needs_label=True)

        queued = unlabelled_clip_paths(self.dir)
        self.assertEqual([p.stem for p in queued], ["todo"])

    def test_relabel_queues_everything(self) -> None:
        self._clip("squat", "done", needs_label=False)
        self._clip("squat", "todo", needs_label=True)

        queued = unlabelled_clip_paths(self.dir, include_labelled=True)
        self.assertEqual(sorted(p.stem for p in queued), ["done", "todo"])


class HoldExerciseTest(unittest.TestCase):
    def test_hold_exercises_are_recognised(self) -> None:
        self.assertTrue(is_hold_exercise("plank"))
        self.assertTrue(is_hold_exercise("wall_sit"))

    def test_rep_exercises_are_not(self) -> None:
        for exercise in ("squat", "pushup", "lunge", "jumping_jack"):
            self.assertFalse(is_hold_exercise(exercise))


if __name__ == "__main__":
    unittest.main()
