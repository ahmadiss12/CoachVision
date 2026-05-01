"""JPEG frame -> MediaPipe pose -> ExerciseDispatcher metrics (v1 WebSocket)."""

from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from coachvision.ai.counters.dispatcher import ExerciseDispatcher
from coachvision.ai.utils.geometry import LandmarkFilter
from coachvision.realtime.contract import WsErrorCode

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
except ImportError:  # pragma: no cover
    mp = None  # type: ignore[assignment]
    vision = None  # type: ignore[assignment]

# Same joint set as fyp1/main.py (normalized names -> MediaPipe indices)
IMPORTANT_LANDMARKS: dict[str, int] = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

_inference_lock = threading.Lock()
_landmarker: vision.PoseLandmarker | None = None
_model_missing_reason: str | None = None


def _model_path() -> Path:
    # coachvision/realtime/pipeline.py -> coachvision/ai/pose_landmarker.task
    return Path(__file__).resolve().parents[1] / "ai" / "pose_landmarker.task"


def _get_landmarker() -> tuple[vision.PoseLandmarker | None, str | None]:
    """Lazy singleton PoseLandmarker; returns (None, reason) if unavailable."""
    global _landmarker, _model_missing_reason
    if _model_missing_reason:
        return None, _model_missing_reason
    if _landmarker is not None:
        return _landmarker, None
    if mp is None or vision is None:
        _model_missing_reason = "mediapipe_not_installed"
        return None, _model_missing_reason
    path = _model_path()
    if not path.is_file():
        _model_missing_reason = f"model_missing:{path}"
        return None, _model_missing_reason
    from mediapipe.tasks import python as mp_python

    base_options = mp_python.BaseOptions(model_asset_path=str(path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    _landmarker = vision.PoseLandmarker.create_from_options(options)
    return _landmarker, None


@dataclass
class LiveSessionState:
    """Per-WebSocket-connection state (no global dispatcher singleton)."""

    session_id: str
    exercise_name: str
    difficulty: str
    dispatcher: ExerciseDispatcher = field(default_factory=ExerciseDispatcher)
    frame_count: int = 0
    fps: float = 30.0
    landmark_filters: dict[str, LandmarkFilter] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dispatcher.set_exercise(self.exercise_name, level=self.difficulty)
        for name in IMPORTANT_LANDMARKS:
            self.landmark_filters[name] = LandmarkFilter(min_cutoff=0.5, beta=0.01)

    def reset_counters(self) -> None:
        self.dispatcher.reset()
        for flt in self.landmark_filters.values():
            flt.reset()
        self.frame_count = 0

    def _extract_landmarks(self, pose_landmarks: Any) -> dict[str, tuple[float, float]]:
        t = time.time()
        out: dict[str, tuple[float, float]] = {}
        lm_list = pose_landmarks  # iterable of landmarks
        for name, idx in IMPORTANT_LANDMARKS.items():
            if idx < len(lm_list):
                lm = lm_list[idx]
                fx, fy = self.landmark_filters[name].filter(lm.x, lm.y, t)
                out[name] = (fx, fy)
        return out


def decode_jpeg_bgr(image_jpeg_base64: str) -> np.ndarray | None:
    raw = base64.b64decode(image_jpeg_base64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def process_jpeg_frame(state: LiveSessionState, image_jpeg_base64: str) -> dict[str, Any]:
    """
    Run one frame. Returns a dict with either:
      - {"kind": "metrics", ...}  teammate metrics shape
      - {"kind": "no_pose", ...}
      - {"kind": "error", "code": str, "message": str}
    """
    landmarker, err = _get_landmarker()
    if landmarker is None:
        return {
            "kind": "error",
            "code": WsErrorCode.MODEL_MISSING,
            "message": err or "pose_model_unavailable",
        }

    frame = decode_jpeg_bgr(image_jpeg_base64)
    if frame is None:
        return {"kind": "error", "code": WsErrorCode.BAD_FRAME, "message": "jpeg_decode_failed"}

    state.frame_count += 1
    timestamp_ms = int(state.frame_count / max(state.fps, 1e-6) * 1000)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    with _inference_lock:
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not detection_result.pose_landmarks:
        return {"kind": "no_pose", "session_id": state.session_id}

    pose_lms = detection_result.pose_landmarks[0]
    landmarks_dict = state._extract_landmarks(pose_lms)
    confidences = [lm.presence for lm in pose_lms]
    confidence = float(np.mean(confidences)) if confidences else 0.0

    count, st, angle = state.dispatcher.update(landmarks_dict, confidence)
    feedback = state.dispatcher.get_feedback()
    progress = state.dispatcher.get_progress()
    state_name = getattr(st, "name", None) or getattr(st, "value", str(st))

    return {
        "kind": "metrics",
        "session_id": state.session_id,
        "count": count,
        "state": state_name,
        "angle": float(angle),
        "feedback": feedback,
        "progress": float(progress),
        "form_name": None,
        "confidence": 0.0,
    }
