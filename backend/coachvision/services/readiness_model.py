"""Learned readiness model (v2): XGBoost regressor over rolling features.

The model predicts the realized session-quality score (0-100, the same
composite recorded by fatigue_calibration) from the pre-workout feature
snapshot. When the trained artifact is missing the model reports
unavailable and callers fall back to the transparent rule engine (v1) —
so shipping this module changes nothing until a model is trained with
scripts/train_readiness_model.py.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - runtime fallback is tested without the package
    xgb = None  # type: ignore[assignment]

from .fatigue_engine import (
    FatiguePredictResult,
    ExplainabilityFactor,
    predict_rule_v1,
    _parse_user_context,
)
from .fatigue_features import RollingFeatureSnapshot

MODEL_VERSION = "xgb_v2"

# Fixed feature order — training script and inference must agree exactly.
READINESS_FEATURE_NAMES = [
    "sessions_7d",
    "reps_7d",
    "avg_form_score_7d",
    "avg_rom_7d",
    "rom_trend_7d",
    "avg_rep_duration_ms_7d",
    "rep_duration_trend_7d",
    "days_since_last_session",
    "form_degradation_7d",
    "shallow_rep_ratio_7d",
    "sleep_hours",
    "soreness",
    "stress",
    "load_ratio",
]


def build_feature_vector(snapshot: dict[str, Any], self_report: dict[str, Any]) -> np.ndarray:
    """Feature vector from serialized snapshot + self-report dicts (NaN = missing)."""

    def _num(value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return parsed if parsed == parsed else float("nan")

    values = [
        _num(snapshot.get("sessions_7d")),
        _num(snapshot.get("reps_7d")),
        _num(snapshot.get("avg_form_score_7d")),
        _num(snapshot.get("avg_rom_7d")),
        _num(snapshot.get("rom_trend_7d")),
        _num(snapshot.get("avg_rep_duration_ms_7d")),
        _num(snapshot.get("rep_duration_trend_7d")),
        _num(snapshot.get("days_since_last_session")),
        _num(snapshot.get("form_degradation_7d")),
        _num(snapshot.get("shallow_rep_ratio_7d")),
        _num(self_report.get("sleepHours", self_report.get("sleep_hours"))),
        _num(self_report.get("muscleSoreness", self_report.get("soreness"))),
        _num(self_report.get("stress")),
        _num(self_report.get("externalLoadBodyweightRatio", self_report.get("load_ratio"))),
    ]
    return np.asarray(values, dtype=np.float32)


class ReadinessModel:
    """Lazy, process-wide loader for the trained readiness regressor."""

    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or Path(__file__).resolve().parent / "models"
        self.model_path = self.model_dir / "readiness_xgboost.json"
        self.metadata_path = self.model_dir / "readiness_xgboost.metadata.json"
        self._booster: Any = None
        self._load_attempted = False
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._booster is not None

    @property
    def load_error(self) -> str | None:
        self._ensure_loaded()
        return self._load_error

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            if xgb is None:
                self._load_error = "xgboost_not_installed"
                return
            if not self.model_path.is_file() or not self.metadata_path.is_file():
                self._load_error = "readiness_model_missing"
                return
            try:
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                if metadata.get("feature_names") != READINESS_FEATURE_NAMES:
                    raise ValueError("model feature order does not match live extractor")
                booster = xgb.Booster()
                booster.load_model(self.model_path)
                self._booster = booster
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                self._load_error = f"readiness_model_load_failed:{exc}"

    def predict_score(self, snapshot: dict[str, Any], self_report: dict[str, Any]) -> int | None:
        self._ensure_loaded()
        if self._booster is None:
            return None
        features = build_feature_vector(snapshot, self_report).reshape(1, -1)
        raw = float(np.asarray(self._booster.inplace_predict(features)).reshape(-1)[0])
        return int(max(0, min(100, round(raw))))


_model = ReadinessModel()


def get_readiness_model() -> ReadinessModel:
    return _model


def _level_for(score: int) -> str:
    if score >= 72:
        return "low"
    if score >= 48:
        return "moderate"
    return "high"


def predict_readiness(
    rolling: RollingFeatureSnapshot,
    user_context: dict[str, Any],
    *,
    window_days: int = 7,
    exercise_id: str | None = None,
) -> tuple[FatiguePredictResult, str]:
    """Best available readiness prediction.

    Always computes the transparent rule result (v1) for explainability.
    If the trained model is available, its score replaces the rule score
    (level/recommendation follow), the rule score is kept in the snapshot
    for A/B comparison, and the returned model version is xgb_v2.
    """
    rule = predict_rule_v1(
        rolling, user_context, window_days=window_days, exercise_id=exercise_id
    )

    model = get_readiness_model()
    if not model.available:
        return rule, "rule_v1"

    snapshot = rule.feature_snapshot
    v2_score = model.predict_score(snapshot, snapshot.get("self_report") or {})
    if v2_score is None:
        return rule, "rule_v1"

    level = _level_for(v2_score)
    recommendations = {
        "low": "Readiness looks solid. You can run your planned session; keep usual rest and technique cues.",
        "moderate": "Reduce planned sets or reps by about 10–15%, extend rest slightly, and stop if form degrades.",
        "high": "Favor a light or mobility-focused day, or switch to an easier variation until recovery improves.",
    }
    snapshot = dict(snapshot)
    snapshot["rule_v1_score"] = rule.readiness_score
    snapshot["xgb_v2_score"] = v2_score

    explain = list(rule.explainability)
    explain.append(
        ExplainabilityFactor(
            key="learned_model",
            label="Learned model",
            impact=v2_score - rule.readiness_score,
            detail=(
                f"Trained model adjusted the rule score {rule.readiness_score} -> {v2_score} "
                "based on how your past sessions actually went."
            ),
        )
    )

    result = FatiguePredictResult(
        readiness_score=v2_score,
        fatigue_level=level,
        recommendation=recommendations[level],
        factors=rule.factors + [f"Learned model: adjusted score to {v2_score} ({v2_score - rule.readiness_score:+d})"],
        explainability=explain,
        feature_snapshot=snapshot,
    )
    return result, MODEL_VERSION
