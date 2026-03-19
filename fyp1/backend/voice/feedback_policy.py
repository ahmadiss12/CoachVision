"""
Feedback policy: outputs a single label per frame (AI/rule-driven).
No queue semantics — one decision per frame, which drives one phrase at most.
"""

from enum import Enum
from typing import Optional, Any


class FeedbackLabel(Enum):
    """Single feedback label per frame. Only one can be active."""
    NONE = "none"
    START_UP = "start_up"
    START_DESCENDING = "start_descending"
    START_ASCENDING = "start_ascending"
    SHALLOW_DEPTH = "shallow_depth"
    TOO_DEEP = "too_deep"
    GOOD_DEPTH = "good_depth"
    REP_COMPLETE = "rep_complete"


class FeedbackPolicy:
    """
    Decides at most one feedback label per frame from state, angle, count, and optional form model.
    Used by the app to drive the VoiceCoach (one phrase at a time, latest wins).
    """

    def __init__(self, exercise_name: str, difficulty: str):
        self.exercise_name = exercise_name.lower()
        self.difficulty = difficulty.lower()
        self._prev_count = 0
        self._prev_state: Optional[Any] = None
        self._bottom_cue_spoken_this_rep = False
        self._rep_cue_spoken_this_rep = False

    def decide(
        self,
        count: int,
        state: Any,
        angle: float,
        form_name: Optional[str] = None,
    ) -> FeedbackLabel:
        """
        Return exactly one label for this frame, or NONE.
        Call once per frame; result drives whether to speak (with cooldown in app).
        """
        if self.exercise_name != "squat":
            return FeedbackLabel.NONE

        state_name = getattr(state, "name", None) or getattr(state, "value", str(state))
        state_upper = state_name.upper() if state_name else ""

        # New rep: reset per-rep flags
        if count > self._prev_count:
            self._bottom_cue_spoken_this_rep = False
            self._rep_cue_spoken_this_rep = False
            self._prev_count = count
            self._prev_state = state
            return FeedbackLabel.REP_COMPLETE

        # Rep complete already spoken this rep — don't repeat
        if count > 0 and self._rep_cue_spoken_this_rep and state_upper == "UP":
            pass  # fall through to state/angle logic

        # In bottom position: at most one depth cue per rep
        if "DOWN" in state_upper and not self._bottom_cue_spoken_this_rep:
            if form_name:
                fn = form_name.lower()
                if "shallow" in fn:
                    self._bottom_cue_spoken_this_rep = True
                    return FeedbackLabel.SHALLOW_DEPTH
                if "too_low" in fn or "too_deep" in fn or "deep" in fn:
                    self._bottom_cue_spoken_this_rep = True
                    return FeedbackLabel.TOO_DEEP
                if "good" in fn or "ok" in fn:
                    self._bottom_cue_spoken_this_rep = True
                    return FeedbackLabel.GOOD_DEPTH
            if angle < 80:
                self._bottom_cue_spoken_this_rep = True
                return FeedbackLabel.TOO_DEEP
            if 90 < angle < 100:
                self._bottom_cue_spoken_this_rep = True
                return FeedbackLabel.SHALLOW_DEPTH
            if 80 <= angle < 90:
                self._bottom_cue_spoken_this_rep = True
                return FeedbackLabel.GOOD_DEPTH

        # State-entry cues: one per transition
        if state != self._prev_state:
            self._prev_state = state
            if "UP" in state_upper:
                return FeedbackLabel.START_UP
            if "DESCENDING" in state_upper:
                return FeedbackLabel.START_DESCENDING
            if "ASCENDING" in state_upper:
                return FeedbackLabel.START_ASCENDING

        return FeedbackLabel.NONE

    def mark_rep_spoken(self) -> None:
        """Call after speaking REP_COMPLETE so we don't repeat."""
        self._rep_cue_spoken_this_rep = True
