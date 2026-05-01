"""
Phrase text per exercise, difficulty, and feedback label.
Used by VoiceCoach to get text for TTS or pre-generated clips.
"""

from typing import Dict
from .feedback_policy import FeedbackLabel


# exercise -> difficulty -> label -> text
PHRASES: Dict[str, Dict[str, Dict[str, str]]] = {
    "squat": {
        "beginner": {
            FeedbackLabel.START_UP.value: "Stand tall, ready to squat",
            FeedbackLabel.START_DESCENDING.value: "Bend your knees and push your hips back",
            FeedbackLabel.START_ASCENDING.value: "Push through your heels and stand up",
            FeedbackLabel.SHALLOW_DEPTH.value: "Go a little deeper",
            FeedbackLabel.TOO_DEEP.value: "Don't go too low",
            FeedbackLabel.GOOD_DEPTH.value: "Good depth",
            FeedbackLabel.REP_COMPLETE.value: "Great rep! Ready for the next one",
        },
        "intermediate": {
            FeedbackLabel.START_UP.value: "Set your stance, feet shoulder-width apart",
            FeedbackLabel.START_DESCENDING.value: "Control the descent, chest up",
            FeedbackLabel.START_ASCENDING.value: "Drive up, full extension",
            FeedbackLabel.SHALLOW_DEPTH.value: "Go deeper",
            FeedbackLabel.TOO_DEEP.value: "Stop at parallel",
            FeedbackLabel.GOOD_DEPTH.value: "Great depth",
            FeedbackLabel.REP_COMPLETE.value: "Solid rep",
        },
        "advanced": {
            FeedbackLabel.START_UP.value: "Set your stance",
            FeedbackLabel.START_DESCENDING.value: "Control the descent",
            FeedbackLabel.START_ASCENDING.value: "Drive up",
            FeedbackLabel.SHALLOW_DEPTH.value: "Hit parallel",
            FeedbackLabel.TOO_DEEP.value: "Maintain form",
            FeedbackLabel.GOOD_DEPTH.value: "Perfect",
            FeedbackLabel.REP_COMPLETE.value: "Good",
        },
    },
}


def get_phrase(exercise: str, difficulty: str, label: FeedbackLabel) -> str:
    """Get phrase text for a label; fallback to beginner if missing."""
    exercise = exercise.lower()
    difficulty = difficulty.lower()
    if exercise not in PHRASES:
        return ""
    diffs = PHRASES[exercise]
    inner = diffs.get(difficulty) or diffs.get("beginner")
    if not inner:
        return ""
    return inner.get(label.value, "")
