"""Tests for deployed XGBoost squat-form feature extraction and smoothing."""

import unittest

import numpy as np

from coachvision.ai.squat_form_classifier import (
    FEATURE_NAMES,
    SquatFormPrediction,
    SquatFormSmoother,
    extract_squat_features,
    get_squat_form_classifier,
)
from coachvision.realtime.pipeline import LiveSessionState, process_pose_landmarks


def sample_landmarks() -> list[list[float]]:
    landmarks = [[0.5, 0.5, 1.0] for _ in range(33)]
    points = {
        11: [0.42, 0.25, 1.0],
        12: [0.58, 0.25, 1.0],
        23: [0.45, 0.50, 1.0],
        24: [0.55, 0.50, 1.0],
        25: [0.44, 0.70, 1.0],
        26: [0.56, 0.70, 1.0],
        27: [0.43, 0.90, 1.0],
        28: [0.57, 0.90, 1.0],
        31: [0.39, 0.94, 1.0],
        32: [0.61, 0.94, 1.0],
    }
    for index, point in points.items():
        landmarks[index] = point
    return landmarks


class SquatFeatureTests(unittest.TestCase):
    def test_extracts_exact_training_feature_count(self):
        features = extract_squat_features(sample_landmarks())
        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(features).all())

    def test_rejects_incomplete_landmark_array(self):
        with self.assertRaises(ValueError):
            extract_squat_features([[0.0, 0.0, 1.0]] * 20)

    def test_probability_smoothing_prevents_single_frame_flip(self):
        smoother = SquatFormSmoother(window_size=3)
        stable = SquatFormPrediction(0, "Correct", 0.9, [0.9, 0.02, 0.02, 0.02, 0.02, 0.02], "")
        outlier = SquatFormPrediction(1, "Shallow", 0.9, [0.02, 0.9, 0.02, 0.02, 0.02, 0.02], "")
        smoother.update(stable)
        smoother.update(stable)
        result = smoother.update(outlier)
        self.assertEqual(result.form_name, "Correct")

    def test_deployed_model_loads_and_returns_six_probabilities(self):
        classifier = get_squat_form_classifier()
        self.assertTrue(classifier.available, classifier.load_error)
        result = classifier.predict(sample_landmarks())
        self.assertIsNotNone(result)
        self.assertEqual(len(result.probabilities), 6)
        self.assertAlmostEqual(sum(result.probabilities), 1.0, places=4)

    def test_live_squat_pipeline_uses_xgboost_without_replacing_counter(self):
        state = LiveSessionState("test-session", "squat", "intermediate")
        result = process_pose_landmarks(state, sample_landmarks(), timestamp_ms=1000)
        self.assertEqual(result["kind"], "metrics")
        self.assertEqual(result["form_source"], "xgboost")
        self.assertIsNotNone(result["form_confidence"])
        self.assertEqual(len(result["form_probabilities"]), 6)
        self.assertIn("count", result)


if __name__ == "__main__":
    unittest.main()
