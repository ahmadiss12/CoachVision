"""Unit tests for rule-based session feedback helpers."""

from __future__ import annotations

import unittest
from uuid import uuid4

from coachvision.db.models import RepEvent
from coachvision.db.models import Session as WorkoutSession
from coachvision.services.session_feedback_generator import (
    build_session_feedback_row,
    export_data_from_rep_events,
    export_data_minimal,
)


class TestSessionFeedbackGenerator(unittest.TestCase):
    def test_build_counts_depth_and_form_issues(self) -> None:
        w = WorkoutSession(user_id=uuid4(), exercise_id="squat", difficulty="intermediate")
        w.id = uuid4()
        w.avg_form_score = 80
        export = {
            "summary": {"total_reps": 2, "good_reps": 1},
            "rep_metrics": [
                {"depth_quality": "shallow"},
                {"form_issues": ["Back rounded"]},
            ],
        }
        fb = build_session_feedback_row(w, export)
        self.assertGreaterEqual(fb.errors_count, 2)
        self.assertEqual(fb.signals_used.get("source"), "live_export_v1")

    def test_export_source_override(self) -> None:
        w = WorkoutSession(user_id=uuid4(), exercise_id="squat", difficulty="intermediate")
        w.id = uuid4()
        fb = build_session_feedback_row(
            w,
            {"summary": {"total_reps": 1}, "rep_metrics": [{"depth_quality": "good"}]},
            export_source="test_v1",
        )
        self.assertEqual(fb.signals_used.get("source"), "test_v1")

    def test_export_minimal_no_reps(self) -> None:
        w = WorkoutSession(user_id=uuid4(), exercise_id="pushup", difficulty="beginner")
        w.total_reps = 0
        data = export_data_minimal(w)
        self.assertEqual(data["summary"]["total_reps"], 0)
        self.assertEqual(data["rep_metrics"], [])

    def test_export_minimal_derives_good_reps_from_avg(self) -> None:
        w = WorkoutSession(user_id=uuid4(), exercise_id="pushup", difficulty="beginner")
        w.total_reps = 10
        w.avg_form_score = 70.0
        data = export_data_minimal(w)
        self.assertEqual(data["summary"]["good_reps"], 7)

    def test_export_from_rep_events_sorted_and_counts(self) -> None:
        sid = uuid4()
        w = WorkoutSession(user_id=uuid4(), exercise_id="squat", difficulty="intermediate")
        r_a = RepEvent(session_id=sid, rep_number=2, depth_quality="shallow")
        r_b = RepEvent(session_id=sid, rep_number=1, depth_quality="good")
        data = export_data_from_rep_events(w, [r_a, r_b])
        self.assertEqual(data["summary"]["total_reps"], 2)
        self.assertEqual(data["summary"]["good_reps"], 1)
        self.assertEqual(data["rep_metrics"][0]["rep_number"], 1)


if __name__ == "__main__":
    unittest.main()
