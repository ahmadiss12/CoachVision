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
from coachvision.ai.voice.feedback_policy import FeedbackLabel, FeedbackPolicy
from coachvision.ai.voice.phrases import get_phrase
from coachvision.realtime.contract import WsErrorCode

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
except ImportError:  # pragma: no cover
    mp = None  # type: ignore[assignment]
    vision = None  # type: ignore[assignment]

# Normalized joint names used by the live exercise counters.
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
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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
    last_timestamp_ms: int = 0
    landmark_filters: dict[str, LandmarkFilter] = field(default_factory=dict)
    voice_policy: FeedbackPolicy = field(init=False)
    last_voice_feedback: str | None = None
    last_voice_count: int = 0

    def __post_init__(self) -> None:
        self.dispatcher.set_exercise(self.exercise_name, level=self.difficulty)
        self.voice_policy = FeedbackPolicy(self.exercise_name, self.difficulty)
        for name in IMPORTANT_LANDMARKS:
            self.landmark_filters[name] = LandmarkFilter(min_cutoff=0.5, beta=0.01)

    def reset_counters(self) -> None:
        self.dispatcher.reset()
        for flt in self.landmark_filters.values():
            flt.reset()
        self.frame_count = 0
        self.voice_policy = FeedbackPolicy(self.exercise_name, self.difficulty)
        self.last_voice_feedback = None
        self.last_voice_count = 0

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


@dataclass(frozen=True)
class ClientPoseLandmark:
    """Small adapter for normalized landmarks produced by client-side MediaPipe JS."""

    x: float
    y: float
    presence: float = 1.0


def _serialize_pose_landmarks(pose_lms: Any) -> list[list[float]]:
    """Normalized [x, y, presence] per landmark for mobile skeleton overlay."""
    return [[float(lm.x), float(lm.y), float(lm.presence)] for lm in pose_lms]


def _infer_form_name(feedback: str | None) -> str:
    if not feedback:
        return "Correct"
    lower = feedback.lower()
    if "shallow" in lower or "deeper" in lower or "depth" in lower:
        return "Shallow"
    if "lean" in lower or "chest" in lower:
        return "Forward Lean"
    return "Adjust"


def _voice_cue_for_metrics(
    state: LiveSessionState,
    count: int,
    movement_state: str,
    angle: float,
    form_name: str | None,
    feedback: str | None,
) -> dict[str, str] | None:
    is_hold_exercise = state.exercise_name.lower().replace("-", "_").replace(" ", "_") in {"plank", "wall_sit"}
    label = state.voice_policy.decide(count, movement_state, angle, form_name)
    if label == FeedbackLabel.NONE:
        phrase = ""
    else:
        if label == FeedbackLabel.REP_COMPLETE:
            state.voice_policy.mark_rep_spoken()
        phrase = get_phrase(state.exercise_name, state.difficulty, label)

    if phrase:
        state.last_voice_count = max(state.last_voice_count, count)
        return {"label": label.value, "text": phrase}

    if count > state.last_voice_count:
        state.last_voice_count = count
        cue_name = "hold_complete" if is_hold_exercise else "rep_complete"
        cue_text = "Good hold" if is_hold_exercise else "Good rep"
        return {"label": f"{cue_name}_{count}", "text": cue_text}

    clean_feedback = (feedback or "").strip()
    if clean_feedback and clean_feedback.lower() not in {"keep going", "keep going."}:
        if clean_feedback != state.last_voice_feedback:
            state.last_voice_feedback = clean_feedback
            return {"label": f"feedback_{abs(hash(clean_feedback))}", "text": clean_feedback}

    state.last_voice_count = max(state.last_voice_count, count)
    return None


def decode_jpeg_bgr(image_jpeg_base64: str) -> np.ndarray | None:
    raw = base64.b64decode(image_jpeg_base64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _resize_for_inference(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def _normalize_timestamp(state: LiveSessionState, timestamp_ms: int | None) -> int:
    state.frame_count += 1
    if timestamp_ms is None:
        timestamp_ms = int(time.monotonic() * 1000)
    if timestamp_ms <= state.last_timestamp_ms:
        timestamp_ms = state.last_timestamp_ms + 1
    state.last_timestamp_ms = timestamp_ms
    return timestamp_ms


def _coerce_client_pose_landmarks(raw_landmarks: Any) -> list[ClientPoseLandmark] | None:
    if not isinstance(raw_landmarks, list):
        return None

    landmarks: list[ClientPoseLandmark] = []
    for raw in raw_landmarks:
        try:
            if isinstance(raw, dict):
                x = float(raw["x"])
                y = float(raw["y"])
                presence = float(raw.get("presence", raw.get("visibility", 1.0)))
            elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
                x = float(raw[0])
                y = float(raw[1])
                presence = float(raw[2]) if len(raw) >= 3 and raw[2] is not None else 1.0
            else:
                return None
        except (KeyError, TypeError, ValueError):
            return None
        landmarks.append(ClientPoseLandmark(x=x, y=y, presence=presence))

    if len(landmarks) <= max(IMPORTANT_LANDMARKS.values()):
        return None
    return landmarks


def _metrics_from_pose_landmarks(
    state: LiveSessionState,
    pose_lms: list[ClientPoseLandmark],
    timestamp_ms: int | None,
    timing_ms: dict[str, float] | None = None,
    total_start: float | None = None,
) -> dict[str, Any]:
    total_start = total_start or time.perf_counter()
    timing_ms = timing_ms or {}
    _normalize_timestamp(state, timestamp_ms)

    landmarks_dict = state._extract_landmarks(pose_lms)
    key_confidences = [
        pose_lms[idx].presence for idx in IMPORTANT_LANDMARKS.values() if idx < len(pose_lms)
    ]
    confidence = float(np.mean(key_confidences)) if key_confidences else 0.0

    count, st, angle = state.dispatcher.update(landmarks_dict, confidence)
    measurement = state.dispatcher.get_live_measurement(count)
    feedback = state.dispatcher.get_feedback()
    progress = state.dispatcher.get_progress()
    state_name = getattr(st, "name", None) or getattr(st, "value", str(st))
    form_name = _infer_form_name(feedback)
    output_feedback = feedback or "Keep going."
    voice = _voice_cue_for_metrics(
        state,
        count,
        state_name,
        float(angle),
        form_name,
        output_feedback,
    )
    timing_ms["total"] = round((time.perf_counter() - total_start) * 1000, 2)

    return {
        "kind": "metrics",
        "session_id": state.session_id,
        "count": measurement["value"],
        "raw_count": count,
        "measurement_type": measurement["type"],
        "measurement_label": measurement["label"],
        "hold_duration_sec": measurement.get("holdDurationSec"),
        "total_hold_time_sec": measurement.get("totalHoldTimeSec"),
        "best_hold_sec": measurement.get("bestHoldSec"),
        "completed_holds": measurement.get("completedHolds"),
        "state": state_name,
        "angle": float(angle),
        "feedback": output_feedback,
        "progress": float(progress),
        "form_name": form_name,
        "confidence": confidence,
        "pose": _serialize_pose_landmarks(pose_lms),
        "voice": voice,
        "timing_ms": timing_ms,
    }


def _process_rgb_image(
    state: LiveSessionState,
    image_rgb: np.ndarray,
    timestamp_ms: int | None = None,
    timing_ms: dict[str, float] | None = None,
    total_start: float | None = None,
) -> dict[str, Any]:
    total_start = total_start or time.perf_counter()
    timing_ms = timing_ms or {}
    landmarker, err = _get_landmarker()
    if landmarker is None:
        return {
            "kind": "error",
            "code": WsErrorCode.MODEL_MISSING,
            "message": err or "pose_model_unavailable",
        }

    timestamp_ms = _normalize_timestamp(state, timestamp_ms)

    prep_start = time.perf_counter()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    timing_ms["prep"] = round((time.perf_counter() - prep_start) * 1000, 2)

    infer_start = time.perf_counter()
    with _inference_lock:
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timing_ms["infer"] = round((time.perf_counter() - infer_start) * 1000, 2)
    timing_ms["total"] = round((time.perf_counter() - total_start) * 1000, 2)

    if not detection_result.pose_landmarks:
        return {
            "kind": "no_pose",
            "session_id": state.session_id,
            "timing_ms": timing_ms,
        }

    pose_lms = detection_result.pose_landmarks[0]
    landmarks_dict = state._extract_landmarks(pose_lms)
    confidences = [lm.presence for lm in pose_lms]
    confidence = float(np.mean(confidences)) if confidences else 0.0

    count, st, angle = state.dispatcher.update(landmarks_dict, confidence)
    measurement = state.dispatcher.get_live_measurement(count)
    feedback = state.dispatcher.get_feedback()
    progress = state.dispatcher.get_progress()
    state_name = getattr(st, "name", None) or getattr(st, "value", str(st))
    form_name = _infer_form_name(feedback)
    output_feedback = feedback or "Keep going."
    voice = _voice_cue_for_metrics(
        state,
        count,
        state_name,
        float(angle),
        form_name,
        output_feedback,
    )

    return {
        "kind": "metrics",
        "session_id": state.session_id,
        "count": measurement["value"],
        "raw_count": count,
        "measurement_type": measurement["type"],
        "measurement_label": measurement["label"],
        "hold_duration_sec": measurement.get("holdDurationSec"),
        "total_hold_time_sec": measurement.get("totalHoldTimeSec"),
        "best_hold_sec": measurement.get("bestHoldSec"),
        "completed_holds": measurement.get("completedHolds"),
        "state": state_name,
        "angle": float(angle),
        "feedback": output_feedback,
        "progress": float(progress),
        "form_name": form_name,
        "confidence": confidence,
        "pose": _serialize_pose_landmarks(pose_lms),
        "voice": voice,
        "timing_ms": timing_ms,
    }


def process_pose_landmarks(
    state: LiveSessionState,
    pose_landmarks: Any,
    timestamp_ms: int | None = None,
    client_inference_ms: float | None = None,
) -> dict[str, Any]:
    """Run counters from client-side pose landmarks without JPEG decode or server inference."""
    total_start = time.perf_counter()
    coerce_start = time.perf_counter()
    pose_lms = _coerce_client_pose_landmarks(pose_landmarks)
    if pose_lms is None:
        return {"kind": "error", "code": WsErrorCode.BAD_FRAME, "message": "bad_pose_landmarks"}

    timing_ms = {
        "coerce": round((time.perf_counter() - coerce_start) * 1000, 2),
        "source": "client_pose",
    }
    if client_inference_ms is not None:
        timing_ms["client_infer"] = round(float(client_inference_ms), 2)

    return _metrics_from_pose_landmarks(
        state,
        pose_lms,
        timestamp_ms,
        timing_ms,
        total_start,
    )


def process_jpeg_frame(
    state: LiveSessionState,
    image_jpeg_base64: str,
    timestamp_ms: int | None = None,
) -> dict[str, Any]:
    """
    Run one JPEG frame. Returns a dict with either:
      - {"kind": "metrics", ...}
      - {"kind": "no_pose", ...}
      - {"kind": "error", "code": str, "message": str}
    """
    total_start = time.perf_counter()
    decode_start = time.perf_counter()
    frame = decode_jpeg_bgr(image_jpeg_base64)
    if frame is None:
        return {"kind": "error", "code": WsErrorCode.BAD_FRAME, "message": "jpeg_decode_failed"}
    timing_ms = {"decode": round((time.perf_counter() - decode_start) * 1000, 2)}

    resize_start = time.perf_counter()
    frame = _resize_for_inference(frame)
    timing_ms["resize"] = round((time.perf_counter() - resize_start) * 1000, 2)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _process_rgb_image(state, image_rgb, timestamp_ms, timing_ms, total_start)
