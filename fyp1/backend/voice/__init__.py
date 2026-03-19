"""
New voice module: single-label feedback policy + VoiceCoach (no queue, latest-wins, instant stop).
"""

from .feedback_policy import FeedbackLabel, FeedbackPolicy
from .voice_coach import VoiceCoach
from .phrases import get_phrase

__all__ = ["FeedbackLabel", "FeedbackPolicy", "VoiceCoach", "get_phrase"]
