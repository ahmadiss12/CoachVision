"""Train the readiness v2 model from accumulated calibration events.

Usage (from backend/, with DATABASE_URL configured in .env):

    python scripts/train_readiness_model.py [--min-samples 150]

Each fatigue_calibration_event that is linked to a pre-workout prediction
provides one training row: X = the prediction's feature snapshot +
self-report, y = the realized session-quality score. The trained model
lands in coachvision/services/models/ and is picked up automatically on
the next backend restart (predictions switch from rule_v1 to xgb_v2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import select  # noqa: E402

from coachvision.db.models import FatigueCalibrationEvent, FatiguePrediction  # noqa: E402
from coachvision.db.session import SessionLocal  # noqa: E402
from coachvision.services.readiness_model import (  # noqa: E402
    READINESS_FEATURE_NAMES,
    build_feature_vector,
)


def load_training_rows():
    db = SessionLocal()
    try:
        rows = db.execute(
            select(FatigueCalibrationEvent, FatiguePrediction)
            .join(FatiguePrediction, FatiguePrediction.id == FatigueCalibrationEvent.prediction_id)
            .where(FatigueCalibrationEvent.prediction_id.is_not(None))
        ).all()
    finally:
        db.close()

    X, y = [], []
    for event, prediction in rows:
        snap = prediction.input_snapshot or {}
        features = snap.get("feature_snapshot") or {}
        self_report = features.get("self_report") or snap.get("userContext") or {}
        X.append(build_feature_vector(features, self_report))
        y.append(float(event.actual_performance_score))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-samples", type=int, default=150)
    args = parser.parse_args()

    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost is not installed — pip install xgboost")
        return 1

    X, y = load_training_rows()
    print(f"Training rows available: {len(y)}")
    if len(y) < args.min_samples:
        print(
            f"Not enough calibration data yet (need >= {args.min_samples}). "
            "Keep training with the app — every completed workout that had a "
            "pre-workout readiness check adds one row."
        )
        return 1

    # Chronology-agnostic random holdout for an honest error estimate.
    rng = np.random.default_rng(42)
    order = rng.permutation(len(y))
    split = int(len(y) * 0.85)
    train_idx, test_idx = order[:split], order[split:]

    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=READINESS_FEATURE_NAMES)
    dtest = xgb.DMatrix(X[test_idx], label=y[test_idx], feature_names=READINESS_FEATURE_NAMES)

    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 4,
        "eval_metric": "mae",
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=400,
        evals=[(dtrain, "train"), (dtest, "holdout")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    holdout_pred = booster.predict(dtest)
    mae = float(np.mean(np.abs(holdout_pred - y[test_idx])))
    print(f"Holdout MAE: {mae:.2f} readiness points over {len(test_idx)} samples")

    out_dir = BACKEND_ROOT / "coachvision" / "services" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "readiness_xgboost.json"
    booster.save_model(model_path)
    (out_dir / "readiness_xgboost.metadata.json").write_text(
        json.dumps(
            {
                "model_version": "xgb_v2",
                "feature_names": READINESS_FEATURE_NAMES,
                "training_rows": int(len(y)),
                "holdout_mae": round(mae, 3),
                "params": params,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {model_path}")
    print("Restart the backend — predictions will now report model_version=xgb_v2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
