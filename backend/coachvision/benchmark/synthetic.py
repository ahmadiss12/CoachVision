"""Generated clips, so the harness can be run and tested without a camera.

**These do not measure counter accuracy.** A synthetic skeleton performs a
textbook rep every time: no occlusion, no camera roll, no half-reps, no bouncing
at the bottom. Scoring near 100% here says the *harness* works, not that the
*counters* do. Real accuracy numbers require recorded clips of real people --
see ``docs/COUNTER_BENCHMARK.md``.

What they are good for:

* proving the replay path end to end before you record anything,
* a fast, deterministic CI smoke test that catches a counter that stops
  counting altogether,
* a worked example of the clip file format.
"""

from __future__ import annotations

import math
import random

from .clips import Frame

# MediaPipe pose landmark indices we populate (see IMPORTANT_LANDMARKS).
LANDMARK_COUNT = 33
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32

# Normalized image coordinates: x right, y *down*.
SIDE_OFFSET = 0.02
"""Horizontal gap between the left and right side of the body in a side view."""

PRESENCE = 0.95
"""Per-landmark presence. The pipeline derives a frame's scalar confidence as
the mean presence of the tracked joints, so the two must agree in a clip."""


def _blank_skeleton() -> list[list[float]]:
    """33 landmarks, all present but parked off-frame until we place them."""
    return [[0.5, 0.5, 0.0] for _ in range(LANDMARK_COUNT)]


def _rotate(vec: tuple[float, float], radians: float) -> tuple[float, float]:
    cos_r, sin_r = math.cos(radians), math.sin(radians)
    return (
        vec[0] * cos_r - vec[1] * sin_r,
        vec[0] * sin_r + vec[1] * cos_r,
    )


def _unit(vec: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vec) or 1.0
    return (vec[0] / length, vec[1] / length)


def _leg_for_knee_angle(knee_angle_deg: float) -> tuple[
    tuple[float, float], tuple[float, float], tuple[float, float]
]:
    """Place hip/knee/ankle so the knee angle is exactly ``knee_angle_deg``.

    The ankle and knee are fixed (the foot stays planted, the shin barely
    moves in a squat); the hip swings to produce the requested angle. That is
    the inverse of what the counter does, which makes the ground truth exact.
    """
    ankle = (0.50, 0.90)
    shin_len, thigh_len = 0.18, 0.20

    # Shin leans slightly forward of vertical, as it does under load.
    shin_lean = 0.15
    knee = (
        ankle[0] + shin_len * math.sin(shin_lean),
        ankle[1] - shin_len * math.cos(shin_lean),
    )

    # Rotate the knee->ankle vector by the target angle; the negative sense puts
    # the hip behind the knee, which is where it belongs in a squat.
    knee_to_ankle = _unit((ankle[0] - knee[0], ankle[1] - knee[1]))
    direction = _rotate(knee_to_ankle, -math.radians(knee_angle_deg))
    hip = (knee[0] + thigh_len * direction[0], knee[1] + thigh_len * direction[1])
    return hip, knee, ankle


def _squat_frame(knee_angle_deg: float, rng: random.Random, noise: float) -> list[list[float]]:
    """One squat pose at a given knee angle, with a little sensor noise."""
    hip, knee, ankle = _leg_for_knee_angle(knee_angle_deg)

    # Torso pitches forward as depth increases -- roughly how people actually squat.
    depth = max(0.0, min(1.0, (170.0 - knee_angle_deg) / 85.0))
    torso_len = 0.26
    torso_lean = 0.12 + 0.45 * depth
    shoulder = (
        hip[0] + torso_len * math.sin(torso_lean),
        hip[1] - torso_len * math.cos(torso_lean),
    )

    lm = _blank_skeleton()

    def place(index: int, point: tuple[float, float], dx: float = 0.0) -> None:
        lm[index] = [
            point[0] + dx + rng.gauss(0.0, noise),
            point[1] + rng.gauss(0.0, noise),
            PRESENCE,
        ]

    for dx, (sh, hp, kn, an, ft) in (
        (0.0, (L_SHOULDER, L_HIP, L_KNEE, L_ANKLE, L_FOOT)),
        (SIDE_OFFSET, (R_SHOULDER, R_HIP, R_KNEE, R_ANKLE, R_FOOT)),
    ):
        place(sh, shoulder, dx)
        place(hp, hip, dx)
        place(kn, knee, dx)
        place(an, ankle, dx)
        place(ft, (ankle[0] + 0.06, ankle[1] + 0.02), dx)

    # Arms held out front for balance; not used by the squat counter, but a
    # realistic clip carries them and other exercises need them.
    place(L_ELBOW, (shoulder[0] + 0.10, shoulder[1] + 0.06))
    place(R_ELBOW, (shoulder[0] + 0.10, shoulder[1] + 0.06), SIDE_OFFSET)
    place(L_WRIST, (shoulder[0] + 0.20, shoulder[1] + 0.04))
    place(R_WRIST, (shoulder[0] + 0.20, shoulder[1] + 0.04), SIDE_OFFSET)
    return lm


def squat_clip(
    reps: int,
    fps: float = 15.0,
    bottom_angle: float = 85.0,
    top_angle: float = 172.0,
    seconds_per_rep: float = 2.4,
    noise: float = 0.0015,
    seed: int = 0,
) -> list[Frame]:
    """A side-view squat clip performing exactly ``reps`` textbook reps.

    Args:
        bottom_angle: Knee angle at the bottom. Above the ``intermediate``
            flexion threshold (90 deg) this becomes a *partial* rep the counter
            is supposed to reject -- useful for testing under-counting.
    """
    rng = random.Random(seed)
    frames: list[Frame] = []
    frames_per_rep = max(4, int(round(seconds_per_rep * fps)))
    standing_pad = int(round(0.8 * fps))

    def emit(angle: float) -> None:
        frames.append(
            Frame(
                t=len(frames) / fps,
                landmarks=_squat_frame(angle, rng, noise),
                confidence=PRESENCE,
            )
        )

    for _ in range(standing_pad):
        emit(top_angle)

    for _ in range(reps):
        for i in range(frames_per_rep):
            # Cosine descent/ascent: smooth, with a natural pause at the bottom.
            phase = (1.0 - math.cos(2.0 * math.pi * i / frames_per_rep)) / 2.0
            emit(top_angle - (top_angle - bottom_angle) * phase)

    for _ in range(standing_pad):
        emit(top_angle)

    return frames


def _plank_frame(
    body_angle_deg: float, rng: random.Random, noise: float
) -> list[list[float]]:
    """One plank pose with a given shoulder-hip-ankle angle (180 = flat)."""
    shoulder = (0.28, 0.56)
    ankle = (0.80, 0.60)

    # Bend the body at the hip by (180 - body_angle): positive sag drops the
    # hips toward the floor (+y), which is the classic plank form break.
    mid = ((shoulder[0] + ankle[0]) / 2.0, (shoulder[1] + ankle[1]) / 2.0)
    sag = math.radians(180.0 - body_angle_deg) * 0.5 * math.hypot(
        ankle[0] - shoulder[0], ankle[1] - shoulder[1]
    )
    hip = (mid[0], mid[1] + sag)

    lm = _blank_skeleton()

    def place(index: int, point: tuple[float, float], dx: float = 0.0) -> None:
        lm[index] = [
            point[0] + dx + rng.gauss(0.0, noise),
            point[1] + rng.gauss(0.0, noise),
            PRESENCE,
        ]

    for dx, (sh, hp, kn, an, ft, el, wr) in (
        (0.0, (L_SHOULDER, L_HIP, L_KNEE, L_ANKLE, L_FOOT, L_ELBOW, L_WRIST)),
        (SIDE_OFFSET, (R_SHOULDER, R_HIP, R_KNEE, R_ANKLE, R_FOOT, R_ELBOW, R_WRIST)),
    ):
        place(sh, shoulder, dx)
        place(hp, hip, dx)
        place(kn, ((hip[0] + ankle[0]) / 2.0, (hip[1] + ankle[1]) / 2.0), dx)
        place(an, ankle, dx)
        place(ft, (ankle[0] + 0.05, ankle[1] + 0.03), dx)
        # Forearms on the floor, directly under the shoulders.
        place(el, (shoulder[0], shoulder[1] + 0.16), dx)
        place(wr, (shoulder[0] + 0.08, shoulder[1] + 0.18), dx)

    return lm


def plank_clip(
    hold_sec: float,
    fps: float = 15.0,
    body_angle: float = 178.0,
    breaks: int = 0,
    break_sec: float = 3.0,
    noise: float = 0.0010,
    seed: int = 0,
) -> list[Frame]:
    """A plank holding good form for ``hold_sec`` seconds total.

    The lead-in and lead-out are below the form threshold, so the counter has to
    find the hold's start and end rather than just timing the whole clip.

    Args:
        breaks: Split the hold into ``breaks + 1`` segments separated by form
            breaks, keeping the total holding time at ``hold_sec``. This is the
            realistic case -- people's hips sag mid-plank -- and it is the case
            that exposes frame-counted (rather than clock-based) hold time.
    """
    rng = random.Random(seed)
    frames: list[Frame] = []
    setup_sec = 1.0
    segments = breaks + 1
    segment_sec = hold_sec / segments

    def emit(angle: float, seconds: float) -> None:
        for _ in range(int(round(seconds * fps))):
            frames.append(
                Frame(
                    t=len(frames) / fps,
                    landmarks=_plank_frame(angle, rng, noise),
                    confidence=PRESENCE,
                )
            )

    # Getting into position: hips clearly sagging, well under the 170 deg gate.
    emit(150.0, setup_sec)
    for segment in range(segments):
        emit(body_angle, segment_sec)
        if segment < segments - 1:
            emit(150.0, break_sec)
    emit(150.0, setup_sec)

    return frames
