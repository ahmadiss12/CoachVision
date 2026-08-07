"""Generate the demo fixture set, so the benchmark runs without a camera.

    python scripts/make_synthetic_clips.py

Writes labelled clips under ``fixtures/``. These validate the harness, not the
counters -- see coachvision/benchmark/synthetic.py for why. Replace them with
real recordings before quoting any accuracy number.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coachvision.benchmark.clips import ClipMeta, write_clip  # noqa: E402
from coachvision.benchmark.synthetic import plank_clip, squat_clip  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def main() -> int:
    written: list[Path] = []

    # Rep clips: same movement, different counts and tempos.
    for clip_id, reps, spr, seed in (
        ("synthetic_squat_05reps", 5, 2.4, 1),
        ("synthetic_squat_08reps", 8, 2.0, 2),
        ("synthetic_squat_12reps_fast", 12, 1.4, 3),
    ):
        written.append(
            write_clip(
                FIXTURES,
                clip_id,
                ClipMeta(
                    exercise="squat",
                    level="intermediate",
                    true_reps=reps,
                    camera_angle="side",
                    source="synthetic",
                    notes="Generated clip. Validates the harness, not counter accuracy.",
                ),
                squat_clip(reps=reps, seconds_per_rep=spr, seed=seed),
            )
        )

    # A partial-depth clip: the knee never passes the intermediate 90 deg gate,
    # so a correct counter scores 0. This is the one that catches a counter
    # tuned so loosely it rewards half-reps.
    written.append(
        write_clip(
            FIXTURES,
            "synthetic_squat_partial_depth",
            ClipMeta(
                exercise="squat",
                level="intermediate",
                true_reps=0,
                camera_angle="side",
                source="synthetic",
                notes="Knee stops at 115 deg -- above the 90 deg gate. Correct answer is 0 reps.",
            ),
            squat_clip(reps=6, bottom_angle=115.0, seed=4),
        )
    )

    # Hold clips. One unbroken, one broken into three segments.
    for clip_id, hold, breaks, seed, note in (
        (
            "synthetic_plank_45sec",
            45.0,
            0,
            6,
            "Generated clip. Validates the harness, not counter accuracy.",
        ),
        (
            "synthetic_plank_60sec_3holds",
            60.0,
            2,
            9,
            "Three 20s holds separated by form breaks. Total holding time is 60s. "
            "Currently FAILS: plank.py accumulates hold time as a hard-coded "
            "1/30s per frame, so at the app's ~15Hz live rate it reports half. "
            "See docs/COUNTER_BENCHMARK.md.",
        ),
    ):
        written.append(
            write_clip(
                FIXTURES,
                clip_id,
                ClipMeta(
                    exercise="plank",
                    level="intermediate",
                    true_hold_sec=hold,
                    camera_angle="side",
                    source="synthetic",
                    notes=note,
                ),
                plank_clip(hold_sec=hold, breaks=breaks, seed=seed),
            )
        )

    for path in written:
        print(f"  wrote {path.relative_to(FIXTURES.parent)}")
    print(f"\n{len(written)} clips in {FIXTURES}")
    print("Now run: python scripts/benchmark_counters.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
