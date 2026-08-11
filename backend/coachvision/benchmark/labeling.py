"""Filling in the ground truth for recorded clips.

The labelling tool never shows a clip's counted result. If the label comes from
the app's own count, every clip scores 100% and the benchmark measures nothing.
What it shows instead is the joint angle over time -- raw geometry, one dip per
repetition -- so a human can count the reps in the recording themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .clips import load_clip

CAMERA_ANGLES = ("side", "front", "45", "other")
HOLD_EXERCISES = {"plank", "wall_sit"}

PLOT_WIDTH = 68
PLOT_HEIGHT = 11

CLIPS_FOR_A_MEANINGFUL_NUMBER = 15
"""Below this a per-exercise rate is noise; see docs/COUNTER_BENCHMARK.md."""


def is_hold_exercise(exercise: str) -> bool:
    return exercise in HOLD_EXERCISES


def angle_plot(
    angles: list[float], width: int = PLOT_WIDTH, height: int = PLOT_HEIGHT
) -> str:
    """Render the joint angle over time so reps can be counted by eye.

    Each repetition is one dip: the joint flexes, then extends again. Buckets
    reduce by *minimum* rather than by average so a fast rep stays a visible
    valley instead of being smoothed into the baseline.
    """
    if len(angles) < 2:
        return "  (not enough frames to plot)"

    bucket = len(angles) / width
    series = [
        min(angles[int(i * bucket) : max(int(i * bucket) + 1, int((i + 1) * bucket))])
        for i in range(width)
    ]

    low, high = min(series), max(series)
    span = high - low
    if span < 1e-6:
        return f"  (joint angle is flat at {low:.0f} deg -- nothing to count)"

    # Row 0 is the largest angle (standing / extended), the last row the smallest.
    def row_of(value: float) -> int:
        return min(height - 1, int((high - value) / span * (height - 1) + 0.5))

    grid = [[" "] * width for _ in range(height)]
    previous = row_of(series[0])
    for column, value in enumerate(series):
        current = row_of(value)
        # Fill between consecutive samples so the trace reads as a continuous
        # line; scattered points make dips much harder to count by eye.
        for row in range(min(previous, current), max(previous, current) + 1):
            grid[row][column] = "#"
        previous = current

    rows = [
        f"  {high:5.0f} |{''.join(grid[0])}" if index == 0 else f"        |{''.join(cells)}"
        for index, cells in enumerate(grid)
    ]
    rows.append(f"  {low:5.0f} +{'-' * width}")
    return "\n".join(rows)


def write_label(
    meta_path: Path,
    *,
    exercise: str,
    value: float,
    camera_angle: str,
    notes: str = "",
) -> dict:
    """Record a human's ground truth into a clip's metadata file."""
    meta = json.loads(meta_path.read_text())

    if is_hold_exercise(exercise):
        meta["true_hold_sec"] = round(float(value), 1)
        meta.pop("true_reps", None)
    else:
        meta["true_reps"] = int(value)
        meta.pop("true_hold_sec", None)

    meta["camera_angle"] = camera_angle
    meta["needs_label"] = False
    if notes:
        meta["notes"] = notes
    else:
        # The recorder leaves labelling instructions here; once labelled they
        # are stale, so an empty answer clears them rather than preserving them.
        meta.pop("notes", None)

    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


@dataclass(frozen=True)
class LabelProgress:
    """How much of one exercise has a ground truth."""

    exercise: str
    labelled: int
    pending: int

    @property
    def total(self) -> int:
        return self.labelled + self.pending

    @property
    def still_needed(self) -> int:
        """Clips short of the point where a rate starts to mean something."""
        return max(0, CLIPS_FOR_A_MEANINGFUL_NUMBER - self.labelled)


def survey(fixtures_dir: Path, exercise: str | None = None) -> list[LabelProgress]:
    """Count labelled and unlabelled clips per exercise."""
    pattern = f"{exercise}/*.jsonl" if exercise else "*/*.jsonl"
    labelled: dict[str, int] = {}
    pending: dict[str, int] = {}

    for path in sorted(fixtures_dir.glob(pattern)):
        clip = load_clip(path)
        bucket = pending if clip.meta.needs_label else labelled
        bucket[clip.meta.exercise] = bucket.get(clip.meta.exercise, 0) + 1

    return [
        LabelProgress(
            exercise=name,
            labelled=labelled.get(name, 0),
            pending=pending.get(name, 0),
        )
        for name in sorted(set(labelled) | set(pending))
    ]


def unlabelled_clip_paths(
    fixtures_dir: Path, exercise: str | None = None, include_labelled: bool = False
) -> list[Path]:
    """Clip files still awaiting a ground truth, in a stable order."""
    pattern = f"{exercise}/*.jsonl" if exercise else "*/*.jsonl"
    return [
        path
        for path in sorted(fixtures_dir.glob(pattern))
        if include_labelled or load_clip(path).meta.needs_label
    ]
