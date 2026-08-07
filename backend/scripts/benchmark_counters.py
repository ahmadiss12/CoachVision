"""Score the exercise counters against labelled clips.

    python scripts/benchmark_counters.py                  # score every clip
    python scripts/benchmark_counters.py --exercise squat # one exercise
    python scripts/benchmark_counters.py --level advanced # override the preset
    python scripts/benchmark_counters.py --per-clip       # show every clip
    python scripts/benchmark_counters.py --check BASELINE.md   # CI regression gate

The regression gate is the point of all this: once a baseline is recorded,
``--check`` fails the build if a threshold change improves one exercise by
breaking another.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coachvision.benchmark import (  # noqa: E402
    aggregate,
    iter_clips,
    markdown_table,
    replay_clip,
    score_clip,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = BACKEND_ROOT / "fixtures"

# "| squat | 4 | 75% | ..." -> ("squat", 0.75)
BASELINE_ROW = re.compile(r"^\|\s*(\w+)\s*\|\s*\d+\s*\|\s*(\d+)%\s*\|")


def parse_baseline(path: Path) -> dict[str, float]:
    """Pull per-exercise exact-rates out of a previously written report."""
    rates: dict[str, float] = {}
    for line in path.read_text().splitlines():
        match = BASELINE_ROW.match(line.strip())
        if match:
            rates[match.group(1)] = int(match.group(2)) / 100.0
    if not rates:
        raise ValueError(f"{path}: no baseline rows found")
    return rates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--exercise", help="Score only this exercise.")
    parser.add_argument(
        "--level",
        choices=("beginner", "intermediate", "advanced"),
        help="Override each clip's own difficulty preset (for threshold sweeps).",
    )
    parser.add_argument("--per-clip", action="store_true", help="List every clip.")
    parser.add_argument("--out", type=Path, help="Write the report here as well.")
    parser.add_argument(
        "--check",
        type=Path,
        metavar="BASELINE.md",
        help="Exit non-zero if any exercise scores below this baseline.",
    )
    args = parser.parse_args()

    clips = list(iter_clips(args.fixtures, args.exercise))
    if not clips:
        print(f"No clips found in {args.fixtures}", file=sys.stderr)
        return 1

    scores = [score_clip(clip, replay_clip(clip, level=args.level)) for clip in clips]

    if args.per_clip:
        print(f"{'clip':<34} {'expected':>9} {'counted':>8} {'error':>7}")
        print("-" * 61)
        for s in sorted(scores, key=lambda s: (s.exercise, s.clip_id)):
            flag = "" if s.exact else "  <-- MISS"
            print(
                f"{s.clip_id:<34} {s.expected:>9g} {s.counted:>8g} {s.error:>+7g}{flag}"
            )
        print()

    summaries = aggregate(scores)
    report = markdown_table(summaries)
    print(report)

    if any(c.meta.source == "synthetic" for c in clips):
        print(
            "\nNote: synthetic clips present. These validate the harness, not "
            "counter accuracy —\nrecord real clips before quoting these numbers."
        )

    if args.out:
        args.out.write_text(report + "\n")
        print(f"\nWrote {args.out}")

    if args.check:
        baseline = parse_baseline(args.check)
        regressions = [
            (s.exercise, baseline[s.exercise], s.exact_rate)
            for s in summaries
            if s.exercise in baseline and s.exact_rate < baseline[s.exercise]
        ]
        if regressions:
            print(f"\nREGRESSION vs {args.check.name}:", file=sys.stderr)
            for exercise, was, now in regressions:
                print(f"  {exercise}: {was:.0%} -> {now:.0%}", file=sys.stderr)
            return 1
        print(f"\nNo regression vs {args.check.name}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
