"""Train the deployed squat-form XGBoost model from extracted pose features."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


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

CLASS_NAMES = [
    "Correct",
    "Shallow",
    "Forward Lean",
    "Knees Caving",
    "Heels Off",
    "Asymmetric",
]

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "objective": "multi:softprob",
    "num_class": 6,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "coachvision" / "ai" / "models",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure_latency_ms(model: XGBClassifier, samples: np.ndarray) -> dict[str, float]:
    for _ in range(20):
        model.predict_proba(samples[:1])

    timings = []
    for row in samples[:1000]:
        started = time.perf_counter()
        model.predict_proba(row.reshape(1, -1))
        timings.append((time.perf_counter() - started) * 1000)

    return {
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(dataset_path)
    missing = [name for name in [*FEATURE_NAMES, "label"] if name not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    # Metadata columns such as source-video frame number are deliberately excluded.
    features = data[FEATURE_NAMES].fillna(0.0).to_numpy(dtype=np.float32)
    labels = data["label"].to_numpy(dtype=np.int64)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = XGBClassifier(**MODEL_PARAMS)
    started = time.perf_counter()
    model.fit(train_x, train_y)
    training_seconds = time.perf_counter() - started

    predictions = model.predict(test_x)
    accuracy = float(accuracy_score(test_y, predictions))
    report = classification_report(
        test_y,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    latency = measure_latency_ms(model, test_x)

    model_path = output_dir / "squat_xgboost.json"
    metadata_path = output_dir / "squat_xgboost.metadata.json"
    model.save_model(model_path)
    metadata = {
        "model_type": "XGBoost multi-class squat-form classifier",
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "training_samples": int(len(train_x)),
        "test_samples": int(len(test_x)),
        "accuracy": accuracy,
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "single_sample_latency": latency,
        "training_seconds": training_seconds,
        "parameters": MODEL_PARAMS,
        "dataset_sha256": file_sha256(dataset_path),
        "notes": (
            "The source-video frame column is excluded because it is metadata and is not "
            "available with equivalent semantics during live inference."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Held-out accuracy: {accuracy:.2%} ({len(test_x)} samples)")
    print(f"Macro F1: {metadata['macro_f1']:.4f}")
    print(f"Inference P95: {latency['p95_ms']:.3f} ms")


if __name__ == "__main__":
    main()
