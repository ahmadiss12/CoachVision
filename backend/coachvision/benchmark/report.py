"""Turn replay results into the numbers that go in the README (and gate CI).

Over-counting and under-counting are reported separately on purpose: they are
different defects. Over-counting means the state machine is retriggering on
jitter or on a partial movement; under-counting means a threshold is too strict
for the range of motion the person actually used. A single "mean error" hides
which one you have, and they need opposite fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .clips import Clip
from .replay import ReplayResult


@dataclass(frozen=True)
class ClipScore:
    """How the counter did on a single clip."""

    clip_id: str
    exercise: str
    is_hold: bool
    expected: float
    counted: float
    camera_angle: str

    @property
    def error(self) -> float:
        """Signed error: positive means the counter over-counted."""
        return self.counted - self.expected

    @property
    def exact(self) -> bool:
        """Hold clips are scored to the nearest 2 seconds, reps must match exactly."""
        return abs(self.error) <= (2.0 if self.is_hold else 0)


@dataclass(frozen=True)
class ExerciseScore:
    """Aggregate across every clip of one exercise."""

    exercise: str
    is_hold: bool
    clips: int
    exact_rate: float
    mae: float
    over_rate: float
    under_rate: float
    worst: ClipScore | None

    @property
    def unit(self) -> str:
        return "sec" if self.is_hold else "reps"


def score_clip(clip: Clip, result: ReplayResult) -> ClipScore:
    """Compare one replay against its ground truth."""
    if clip.meta.is_hold:
        expected: float = float(clip.meta.true_hold_sec or 0.0)
        counted: float = result.counted_hold_sec
    else:
        expected = float(clip.meta.true_reps or 0)
        counted = float(result.counted_reps)

    return ClipScore(
        clip_id=clip.clip_id,
        exercise=clip.meta.exercise,
        is_hold=clip.meta.is_hold,
        expected=expected,
        counted=counted,
        camera_angle=clip.meta.camera_angle,
    )


def aggregate(scores: list[ClipScore]) -> list[ExerciseScore]:
    """Roll per-clip scores up per exercise, ordered worst-accuracy first."""
    by_exercise: dict[str, list[ClipScore]] = {}
    for score in scores:
        by_exercise.setdefault(score.exercise, []).append(score)

    out: list[ExerciseScore] = []
    for exercise, group in by_exercise.items():
        n = len(group)
        # "Over" and "under" only count clips that were actually wrong, so the
        # three rates (exact + over + under) sum to 1.
        over = sum(1 for s in group if not s.exact and s.error > 0)
        under = sum(1 for s in group if not s.exact and s.error < 0)
        out.append(
            ExerciseScore(
                exercise=exercise,
                is_hold=group[0].is_hold,
                clips=n,
                exact_rate=sum(1 for s in group if s.exact) / n,
                mae=sum(abs(s.error) for s in group) / n,
                over_rate=over / n,
                under_rate=under / n,
                worst=max(group, key=lambda s: abs(s.error)),
            )
        )

    out.sort(key=lambda e: (e.exact_rate, -e.mae))
    return out


def markdown_table(summaries: list[ExerciseScore]) -> str:
    """Render the aggregate table, worst exercise first."""
    lines = [
        "| Exercise | Clips | Exact | MAE | Over | Under | Worst clip |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in summaries:
        worst = (
            f"{s.worst.clip_id} ({s.worst.counted:g} vs {s.worst.expected:g})"
            if s.worst and not s.worst.exact
            else "-"
        )
        lines.append(
            f"| {s.exercise} | {s.clips} | {s.exact_rate:.0%} | "
            f"{s.mae:.2f} {s.unit} | {s.over_rate:.0%} | {s.under_rate:.0%} | {worst} |"
        )

    total = sum(s.clips for s in summaries)
    if total:
        overall = sum(s.exact_rate * s.clips for s in summaries) / total
        lines.append("")
        lines.append(f"**Overall: {overall:.1%} exact across {total} clips.**")
    return "\n".join(lines)
