"""Unit tests for the rule-based fatigue engine (v1)."""

import unittest

from coachvision.services.fatigue_engine import predict_rule_v1
from coachvision.services.fatigue_features import RollingFeatureSnapshot


def _snapshot(**overrides) -> RollingFeatureSnapshot:
    base = dict(
        sessions_7d=0,
        reps_7d=0,
        avg_form_score_7d=None,
        avg_rom_7d=None,
        rom_trend_7d=None,
        avg_rep_duration_ms_7d=None,
        days_since_last_session=2,
        form_degradation_7d=None,
        shallow_rep_ratio_7d=None,
    )
    base.update(overrides)
    return RollingFeatureSnapshot(**base)


class FatigueEngineTest(unittest.TestCase):
    def test_window_size_does_not_change_score_for_same_weekly_rate(self) -> None:
        """4 sessions/120 reps over 7 days == 8 sessions/240 reps over 14 days."""
        weekly = predict_rule_v1(
            _snapshot(sessions_7d=4, reps_7d=120),
            {},
            window_days=7,
            exercise_id="squat",
        )
        biweekly = predict_rule_v1(
            _snapshot(sessions_7d=8, reps_7d=240),
            {},
            window_days=14,
            exercise_id="squat",
        )
        self.assertEqual(weekly.readiness_score, biweekly.readiness_score)

    def test_higher_frequency_scores_lower(self) -> None:
        light = predict_rule_v1(_snapshot(sessions_7d=1), {}, window_days=7, exercise_id="squat")
        heavy = predict_rule_v1(_snapshot(sessions_7d=6), {}, window_days=7, exercise_id="squat")
        self.assertGreater(light.readiness_score, heavy.readiness_score)

    def test_garbage_user_context_does_not_crash(self) -> None:
        result = predict_rule_v1(
            _snapshot(),
            {
                "sleepHours": "seven-ish",
                "muscleSoreness": None,
                "stress": "very",
                "externalLoadKg": "heavy",
                "bodyWeightKg": float("nan"),
            },
            window_days=7,
            exercise_id="squat",
        )
        self.assertTrue(0 <= result.readiness_score <= 100)
        self.assertIn(result.fatigue_level, {"low", "moderate", "high"})

    def test_absurd_sleep_is_clamped(self) -> None:
        normal = predict_rule_v1(_snapshot(), {"sleepHours": 8}, window_days=7, exercise_id="squat")
        absurd = predict_rule_v1(_snapshot(), {"sleepHours": 9999}, window_days=7, exercise_id="squat")
        self.assertEqual(normal.readiness_score, absurd.readiness_score)

    def test_external_load_only_penalizes_load_bearing_exercises(self) -> None:
        ctx = {"externalLoadKg": 40, "bodyWeightKg": 80}
        squat = predict_rule_v1(_snapshot(), ctx, window_days=7, exercise_id="squat")
        pushup = predict_rule_v1(_snapshot(), ctx, window_days=7, exercise_id="pushup")
        self.assertLess(squat.readiness_score, pushup.readiness_score)
        load_keys = {factor.key for factor in pushup.explainability}
        self.assertNotIn("external_load", load_keys)

    def test_poor_recovery_signals_lower_score_and_level(self) -> None:
        rested = predict_rule_v1(
            _snapshot(days_since_last_session=3),
            {"sleepHours": 8, "muscleSoreness": 1, "stress": 1},
            window_days=7,
            exercise_id="squat",
        )
        wrecked = predict_rule_v1(
            _snapshot(sessions_7d=6, reps_7d=400, days_since_last_session=0, shallow_rep_ratio_7d=0.5),
            {"sleepHours": 4, "muscleSoreness": 5, "stress": 5},
            window_days=7,
            exercise_id="squat",
        )
        self.assertEqual(rested.fatigue_level, "low")
        self.assertEqual(wrecked.fatigue_level, "high")
        self.assertGreater(rested.readiness_score, wrecked.readiness_score + 30)

    def test_slowing_rep_tempo_penalized(self) -> None:
        steady = predict_rule_v1(
            _snapshot(rep_duration_trend_7d=1.0), {}, window_days=7, exercise_id="squat"
        )
        slowing = predict_rule_v1(
            _snapshot(rep_duration_trend_7d=1.3), {}, window_days=7, exercise_id="squat"
        )
        self.assertGreater(steady.readiness_score, slowing.readiness_score)
        keys = {factor.key for factor in slowing.explainability}
        self.assertIn("rep_tempo_trend", keys)

    def test_readiness_v2_falls_back_to_rule_without_model(self) -> None:
        from coachvision.services.readiness_model import predict_readiness

        result, version = predict_readiness(
            _snapshot(), {"sleepHours": 8}, window_days=7, exercise_id="squat"
        )
        self.assertEqual(version, "rule_v1")
        rule = predict_rule_v1(
            _snapshot(), {"sleepHours": 8}, window_days=7, exercise_id="squat"
        )
        self.assertEqual(result.readiness_score, rule.readiness_score)

    def test_feature_vector_handles_missing_values(self) -> None:
        import numpy as np

        from coachvision.services.readiness_model import (
            READINESS_FEATURE_NAMES,
            build_feature_vector,
        )

        vector = build_feature_vector({}, {})
        self.assertEqual(len(vector), len(READINESS_FEATURE_NAMES))
        self.assertTrue(np.isnan(vector).all())

        vector = build_feature_vector(
            {"sessions_7d": 3, "avg_form_score_7d": "not-a-number"},
            {"sleepHours": 7.5},
        )
        self.assertEqual(vector[0], 3.0)
        self.assertTrue(np.isnan(vector[2]))
        self.assertEqual(vector[10], 7.5)

    def test_every_factor_is_explained(self) -> None:
        result = predict_rule_v1(
            _snapshot(sessions_7d=5, reps_7d=300, avg_form_score_7d=70, shallow_rep_ratio_7d=0.4),
            {"sleepHours": 5, "muscleSoreness": 4, "stress": 4, "externalLoadKg": 30, "bodyWeightKg": 75},
            window_days=7,
            exercise_id="squat",
        )
        # Sum of impacts must reconcile with the final score (base 78, clamped 0-100).
        total_impact = sum(factor.impact for factor in result.explainability)
        self.assertEqual(result.readiness_score, max(0, min(100, 78 + total_impact)))


if __name__ == "__main__":
    unittest.main()
