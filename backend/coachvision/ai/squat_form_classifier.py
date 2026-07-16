"""Live squat-form classification from MediaPipe landmarks using XGBoost."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - runtime fallback is tested without the package
    xgb = None  # type: ignore[assignment]


FEATURE_NAMES = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "spine_angle",
    "torso_lean",
    "left_knee_lateral",
    "right_knee_lateral",
    "symmetry_score",
    "hip_depth",
]

DEFAULT_CLASS_NAMES = [
    "Correct",
    "Shallow",
    "Forward Lean",
    "Knees Caving",
    "Heels Off",
    "Asymmetric",
]

FORM_FEEDBACK = {
    "Correct": "Good form. Keep going.",
    "Shallow": "Go deeper to complete the squat.",
    "Forward Lean": "Keep your chest up and reduce forward lean.",
    "Knees Caving": "Push your knees outward in line with your feet.",
    "Heels Off": "Keep both heels flat on the floor.",
    "Asymmetric": "Balance your weight evenly on both sides.",
}


@dataclass(frozen=True)
class SquatFormPrediction:
    form_id: int
    form_name: str
    confidence: float
    probabilities: list[float]
    feedback: str
    inference_ms: float = 0.0


def _xy(landmarks: Sequence[Any], index: int) -> tuple[float, float]:
    landmark = landmarks[index]
    if isinstance(landmark, dict):
        return float(landmark["x"]), float(landmark["y"])
    if isinstance(landmark, (list, tuple, np.ndarray)):
        return float(landmark[0]), float(landmark[1])
    return float(landmark.x), float(landmark.y)


def _angle(first: tuple[float, float], vertex: tuple[float, float], third: tuple[float, float]) -> float:
    left = np.asarray(first, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    right = np.asarray(third, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-9:
        return 0.0
    cosine = float(np.dot(left, right) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def extract_squat_features(landmarks: Sequence[Any]) -> np.ndarray:
    """Return the exact 12-feature vector used to train the deployed model."""
    if len(landmarks) < 33:
        raise ValueError("Expected 33 MediaPipe pose landmarks")

    left_shoulder = _xy(landmarks, 11)
    right_shoulder = _xy(landmarks, 12)
    left_hip = _xy(landmarks, 23)
    right_hip = _xy(landmarks, 24)
    left_knee = _xy(landmarks, 25)
    right_knee = _xy(landmarks, 26)
    left_ankle = _xy(landmarks, 27)
    right_ankle = _xy(landmarks, 28)
    left_foot = _xy(landmarks, 31)
    right_foot = _xy(landmarks, 32)

    left_knee_angle = _angle(left_hip, left_knee, left_ankle)
    right_knee_angle = _angle(right_hip, right_knee, right_ankle)
    left_hip_angle = _angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = _angle(right_shoulder, right_hip, right_knee)
    left_ankle_angle = _angle(left_knee, left_ankle, left_foot)
    right_ankle_angle = _angle(right_knee, right_ankle, right_foot)
    spine_angle = _angle(left_shoulder, left_hip, right_hip)
    torso_lean = abs(left_shoulder[0] - left_hip[0])
    left_knee_lateral = abs(left_knee[0] - left_hip[0])
    right_knee_lateral = abs(right_knee[0] - right_hip[0])
    symmetry_score = abs(left_knee_angle - right_knee_angle)
    hip_depth = left_hip[1] - left_shoulder[1]

    features = np.asarray(
        [
            left_knee_angle,
            right_knee_angle,
            left_hip_angle,
            right_hip_angle,
            left_ankle_angle,
            right_ankle_angle,
            spine_angle,
            torso_lean,
            left_knee_lateral,
            right_knee_lateral,
            symmetry_score,
            hip_depth,
        ],
        dtype=np.float32,
    )
    if not np.isfinite(features).all():
        raise ValueError("Squat features contain non-finite values")
    return features


class SquatFormClassifier:
    """Lazy, process-wide XGBoost model loader and predictor."""

    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or Path(__file__).resolve().parent / "models"
        self.model_path = self.model_dir / "squat_xgboost.json"
        self.metadata_path = self.model_dir / "squat_xgboost.metadata.json"
        self._booster: Any = None
        self._class_names = DEFAULT_CLASS_NAMES
        self._load_attempted = False
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @property
    def load_error(self) -> str | None:
        self._ensure_loaded()
        return self._load_error

    @property
    def class_names(self) -> list[str]:
        self._ensure_loaded()
        return self._class_names

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._booster is not None

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        with self._load_lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            if xgb is None:
                self._load_error = "xgboost_not_installed"
                return
            if not self.model_path.is_file() or not self.metadata_path.is_file():
                self._load_error = "squat_xgboost_model_missing"
                return
            try:
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                if metadata.get("feature_names") != FEATURE_NAMES:
                    raise ValueError("model feature order does not match live extractor")
                self._class_names = list(metadata.get("class_names") or DEFAULT_CLASS_NAMES)
                booster = xgb.Booster()
                booster.load_model(self.model_path)
                self._booster = booster
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                self._load_error = f"squat_xgboost_load_failed:{exc}"

    def predict(self, landmarks: Sequence[Any]) -> SquatFormPrediction | None:
        self._ensure_loaded()
        if self._booster is None or xgb is None:
            return None

        features = extract_squat_features(landmarks)
        started = time.perf_counter()
        with self._predict_lock:
            raw = self._booster.inplace_predict(features.reshape(1, -1))
        inference_ms = (time.perf_counter() - started) * 1000
        probabilities = np.asarray(raw, dtype=np.float64).reshape(-1)
        form_id = int(np.argmax(probabilities))
        form_name = self._class_names[form_id]
        return SquatFormPrediction(
            form_id=form_id,
            form_name=form_name,
            confidence=float(probabilities[form_id]),
            probabilities=probabilities.tolist(),
            feedback=FORM_FEEDBACK.get(form_name, "Adjust your squat form."),
            inference_ms=inference_ms,
        )


class SquatFormSmoother:
    """Average recent class probabilities to avoid frame-to-frame label flicker.

    Adds label hysteresis on top of the moving average: once a label is
    emitted, a different label must beat it by ``switch_margin`` (smoothed
    probability) before the emitted label changes. This prevents rapid
    label/voice-cue flapping when two classes are nearly tied.
    """

    def __init__(self, window_size: int = 7, switch_margin: float = 0.05):
        self._history: deque[np.ndarray] = deque(maxlen=window_size)
        self._switch_margin = switch_margin
        self._current_id: int | None = None

    def reset(self) -> None:
        self._history.clear()
        self._current_id = None

    def update(
        self,
        prediction: SquatFormPrediction,
        class_names: Sequence[str] | None = None,
    ) -> SquatFormPrediction:
        names = list(class_names) if class_names else DEFAULT_CLASS_NAMES
        self._history.append(np.asarray(prediction.probabilities, dtype=np.float64))
        probabilities = np.mean(np.stack(self._history), axis=0)
        best_id = int(np.argmax(probabilities))

        if (
            self._current_id is not None
            and best_id != self._current_id
            and self._current_id < len(probabilities)
            and probabilities[best_id] - probabilities[self._current_id] < self._switch_margin
        ):
            best_id = self._current_id
        self._current_id = best_id

        form_name = names[best_id] if best_id < len(names) else str(best_id)
        return SquatFormPrediction(
            form_id=best_id,
            form_name=form_name,
            confidence=float(probabilities[best_id]),
            probabilities=probabilities.tolist(),
            feedback=FORM_FEEDBACK.get(form_name, "Adjust your squat form."),
            inference_ms=prediction.inference_ms,
        )


_classifier = SquatFormClassifier()


def get_squat_form_classifier() -> SquatFormClassifier:
    return _classifier
