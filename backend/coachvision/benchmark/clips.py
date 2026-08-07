"""On-disk format for benchmark clips.

A clip is a pair of files under ``fixtures/<exercise>/``:

``<clip_id>.jsonl``
    One JSON object per frame, in capture order::

        {"t": 0.067, "confidence": 0.94, "landmarks": [[x, y, presence], ...]}

    ``t`` is seconds since the start of the clip, and ``landmarks`` is the same
    ``[x, y, presence]`` array the WebSocket already uses for the ``pose``
    message (see WS_CONTRACT.md), so a recorder is a straight dump of what the
    client sends -- no conversion, no video.

``<clip_id>.meta.json``
    The ground truth label, written by a human who watched the clip::

        {"exercise": "squat", "level": "intermediate", "true_reps": 5}

    Hold exercises (plank, wall sit) carry ``true_hold_sec`` instead.

Landmark coordinates are normalized floats, not pixels, so a clip contains no
image data and nothing personally identifiable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class Frame:
    """One captured pose frame."""

    t: float
    """Seconds since the start of the clip."""

    landmarks: list[list[float]]
    """Per-landmark ``[x, y, presence]``, MediaPipe ordering."""

    confidence: float = 1.0
    """Overall pose confidence the client reported for this frame."""


@dataclass(frozen=True)
class ClipMeta:
    """Ground truth for a clip -- what a human says actually happened."""

    exercise: str
    level: str = "intermediate"
    true_reps: int | None = None
    true_hold_sec: float | None = None
    camera_angle: str = "unknown"
    source: str = "recorded"
    notes: str = ""

    @property
    def is_hold(self) -> bool:
        return self.true_hold_sec is not None

    def __post_init__(self) -> None:
        if self.true_reps is None and self.true_hold_sec is None:
            raise ValueError(
                f"{self.exercise}: clip metadata must set either true_reps or true_hold_sec"
            )


@dataclass(frozen=True)
class Clip:
    """A labelled landmark recording."""

    clip_id: str
    meta: ClipMeta
    frames: list[Frame] = field(repr=False)

    @property
    def duration_sec(self) -> float:
        if not self.frames:
            return 0.0
        return self.frames[-1].t - self.frames[0].t

    @property
    def fps(self) -> float:
        if self.duration_sec <= 0:
            return 0.0
        return (len(self.frames) - 1) / self.duration_sec


def load_clip(jsonl_path: Path) -> Clip:
    """Load a clip and its sidecar metadata."""
    meta_path = jsonl_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path.name} has no ground truth: expected {meta_path.name} beside it"
        )

    raw_meta: dict[str, Any] = json.loads(meta_path.read_text())
    # clip_id is derived from the filename, so a copied fixture can't silently
    # keep a stale id from the metadata.
    raw_meta.pop("clip_id", None)
    meta = ClipMeta(**raw_meta)

    frames: list[Frame] = []
    with jsonl_path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path.name}:{lineno}: bad JSON ({exc})") from exc
            frames.append(
                Frame(
                    t=float(row["t"]),
                    landmarks=row["landmarks"],
                    confidence=float(row.get("confidence", 1.0)),
                )
            )

    if not frames:
        raise ValueError(f"{jsonl_path.name}: clip has no frames")

    return Clip(clip_id=jsonl_path.stem, meta=meta, frames=frames)


def iter_clips(fixtures_dir: Path, exercise: str | None = None) -> Iterator[Clip]:
    """Yield every labelled clip under ``fixtures_dir``, sorted for stable reports."""
    if not fixtures_dir.exists():
        raise FileNotFoundError(
            f"No fixtures at {fixtures_dir}. Generate the demo set with:\n"
            f"  python scripts/make_synthetic_clips.py"
        )

    pattern = f"{exercise}/*.jsonl" if exercise else "*/*.jsonl"
    for path in sorted(fixtures_dir.glob(pattern)):
        yield load_clip(path)


def write_clip(fixtures_dir: Path, clip_id: str, meta: ClipMeta, frames: list[Frame]) -> Path:
    """Write a clip to disk in the format :func:`load_clip` expects."""
    out_dir = fixtures_dir / meta.exercise
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / f"{clip_id}.jsonl"
    with jsonl_path.open("w") as fh:
        for frame in frames:
            fh.write(
                json.dumps(
                    {
                        "t": round(frame.t, 4),
                        "confidence": round(frame.confidence, 3),
                        "landmarks": [[round(v, 5) for v in lm] for lm in frame.landmarks],
                    }
                )
                + "\n"
            )

    meta_payload = {k: v for k, v in vars(meta).items() if v is not None}
    (out_dir / f"{clip_id}.meta.json").write_text(json.dumps(meta_payload, indent=2) + "\n")
    return jsonl_path
