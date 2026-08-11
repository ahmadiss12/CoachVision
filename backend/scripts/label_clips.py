"""Fill in the ground truth for recorded clips.

    python scripts/label_clips.py            # walk through everything unlabelled
    python scripts/label_clips.py --status   # what is labelled, what is not
    python scripts/label_clips.py --exercise squat
    python scripts/label_clips.py --relabel  # revisit clips already labelled

For each clip it plots the movement so you can count the reps from the
recording itself, then asks what actually happened.

It never shows you what the counter said. That is deliberate: if the label
comes from the app's own count, every clip scores 100% and the benchmark
measures nothing. The plot is raw joint geometry -- one dip per repetition --
not the counter's decision about it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coachvision.benchmark.clips import load_clip  # noqa: E402
from coachvision.benchmark.labeling import (  # noqa: E402
    CAMERA_ANGLES,
    CLIPS_FOR_A_MEANINGFUL_NUMBER,
    angle_plot,
    is_hold_exercise,
    survey,
    unlabelled_clip_paths,
    write_label,
)
from coachvision.benchmark.replay import replay_clip  # noqa: E402

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = BACKEND_ROOT / "fixtures"


def _ask_count(prompt: str) -> float | str:
    """Read a number, or a control word. Re-asks until the input makes sense."""
    while True:
        raw = input(f"  {prompt}\n  > ").strip().lower()
        if raw in ("s", "skip"):
            return "skip"
        if raw in ("q", "quit"):
            return "quit"
        try:
            value = float(raw)
        except ValueError:
            print("    Enter a number, 's' to skip this clip, or 'q' to stop.")
            continue
        if value < 0:
            print("    Cannot be negative.")
            continue
        return value


def _ask_choice(prompt: str, options: tuple[str, ...], default: str) -> str:
    raw = input(f"  {prompt} [{'/'.join(options)}] (default {default}): ").strip().lower()
    if not raw:
        return default
    for option in options:
        if option.startswith(raw):
            return option
    return raw


def label_one(jsonl_path: Path, index: int, total: int) -> str:
    """Label a single clip. Returns 'labelled', 'skip', or 'quit'."""
    clip = load_clip(jsonl_path)
    is_hold = is_hold_exercise(clip.meta.exercise)

    print(f"\n{'=' * 78}")
    print(f"Clip {index} of {total}: {clip.clip_id}")
    print(f"  Exercise : {clip.meta.exercise} ({'hold' if is_hold else 'reps'})")
    print(
        f"  Recorded : {clip.duration_sec:.1f}s, "
        f"{len(clip.frames)} frames, {clip.fps:.1f} fps"
    )
    print()

    # Raw geometry only -- the counter's own tally is never printed here.
    result = replay_clip(clip)
    caption = "one plateau = one hold" if is_hold else "one dip = one repetition"
    print(f"  Joint angle over time ({caption}):")
    print(angle_plot(result.angles))
    print()

    answer = _ask_count(
        "How many SECONDS did you hold good form? (s=skip, q=quit)"
        if is_hold
        else "How many reps did you actually do? (s=skip, q=quit)"
    )
    if isinstance(answer, str):
        return answer

    camera = _ask_choice("Camera angle?", CAMERA_ANGLES, clip.meta.camera_angle or "side")
    notes = input("  Notes (optional): ").strip()

    write_label(
        jsonl_path.with_suffix(".meta.json"),
        exercise=clip.meta.exercise,
        value=answer,
        camera_angle=camera,
        notes=notes,
    )
    print(f"  Saved: {answer:g} {'sec' if is_hold else 'reps'}, {camera} view.")
    return "labelled"


def show_status(fixtures: Path, exercise: str | None) -> int:
    progress = survey(fixtures, exercise)
    if not progress:
        print(f"No clips in {fixtures}")
        return 1

    print(f"{'Exercise':<20} {'Labelled':>9} {'Pending':>8}   Progress")
    print("-" * 66)
    for row in progress:
        bar = "#" * min(CLIPS_FOR_A_MEANINGFUL_NUMBER, row.labelled)
        bar += "." * max(
            0, min(CLIPS_FOR_A_MEANINGFUL_NUMBER, row.total) - row.labelled
        )
        short = f"  (need {row.still_needed} more)" if row.still_needed else "  ready"
        print(f"{row.exercise:<20} {row.labelled:>9} {row.pending:>8}   {bar}{short}")

    pending = sum(row.pending for row in progress)
    print()
    if pending:
        print(f"{pending} clip(s) still need a label. Run: python scripts/label_clips.py")
    else:
        print("Everything is labelled. Run: python scripts/benchmark_counters.py --per-clip")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--exercise", help="Only this exercise.")
    parser.add_argument("--status", action="store_true", help="Show progress and exit.")
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Also revisit clips that already have a label.",
    )
    args = parser.parse_args()

    if not args.fixtures.exists():
        print(f"No fixtures at {args.fixtures}.", file=sys.stderr)
        print("Record some workouts first -- see docs/COUNTER_BENCHMARK.md.", file=sys.stderr)
        return 1

    if args.status:
        return show_status(args.fixtures, args.exercise)

    todo = unlabelled_clip_paths(args.fixtures, args.exercise, include_labelled=args.relabel)
    if not todo:
        print("Nothing to label. Run: python scripts/benchmark_counters.py --per-clip")
        return 0

    print(f"{len(todo)} clip(s) to label.")
    print("Count from the plot, not from what the app showed you during the workout.")

    done = 0
    for index, path in enumerate(todo, start=1):
        try:
            outcome = label_one(path, index, len(todo))
        except (KeyboardInterrupt, EOFError):
            print("\nStopped.")
            break
        if outcome == "quit":
            print("\nStopped.")
            break
        if outcome == "labelled":
            done += 1

    print(f"\nLabelled {done} of {len(todo)}.")
    if done:
        print("Now run: python scripts/benchmark_counters.py --per-clip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
