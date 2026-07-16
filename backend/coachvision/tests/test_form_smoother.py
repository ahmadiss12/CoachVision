"""Tests for the squat-form probability smoother (class names + hysteresis)."""

import unittest

from coachvision.ai.squat_form_classifier import SquatFormPrediction, SquatFormSmoother


def _prediction(probabilities: list[float]) -> SquatFormPrediction:
    form_id = probabilities.index(max(probabilities))
    return SquatFormPrediction(
        form_id=form_id,
        form_name=str(form_id),
        confidence=max(probabilities),
        probabilities=probabilities,
        feedback="",
    )


class FormSmootherTest(unittest.TestCase):
    def test_uses_model_class_names_not_defaults(self) -> None:
        """A retrained model may reorder classes; the smoother must honor it."""
        custom = ["Knees Caving", "Correct", "Shallow", "Forward Lean", "Heels Off", "Asymmetric"]
        smoother = SquatFormSmoother(window_size=3)
        result = smoother.update(_prediction([0.9, 0.02, 0.02, 0.02, 0.02, 0.02]), custom)
        self.assertEqual(result.form_name, "Knees Caving")
        self.assertEqual(result.feedback, "Push your knees outward in line with your feet.")

    def test_hysteresis_prevents_label_flapping(self) -> None:
        smoother = SquatFormSmoother(window_size=2, switch_margin=0.15)
        first = smoother.update(_prediction([0.60, 0.40, 0, 0, 0, 0]))
        self.assertEqual(first.form_id, 0)

        # Smoothed probabilities are now [0.45, 0.55]: class 1 leads, but only
        # by 0.10 — inside the 0.15 margin, so the label should hold.
        second = smoother.update(_prediction([0.30, 0.70, 0, 0, 0, 0]))
        self.assertEqual(second.form_id, 0)

        # A decisive shift must switch the label.
        third = smoother.update(_prediction([0.05, 0.95, 0, 0, 0, 0]))
        self.assertEqual(third.form_id, 1)

    def test_reset_clears_hysteresis_state(self) -> None:
        smoother = SquatFormSmoother(window_size=2, switch_margin=0.10)
        smoother.update(_prediction([0.9, 0.1, 0, 0, 0, 0]))
        smoother.reset()
        result = smoother.update(_prediction([0.1, 0.9, 0, 0, 0, 0]))
        self.assertEqual(result.form_id, 1)


if __name__ == "__main__":
    unittest.main()
