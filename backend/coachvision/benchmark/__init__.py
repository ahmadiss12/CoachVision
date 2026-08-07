"""Counter accuracy benchmark: replay recorded landmark clips, score the counters.

The exercise counters are pure functions of a landmark sequence -- no camera and
no MediaPipe are needed to exercise them. A "clip" here is therefore just the
stream of landmarks the mobile app already sends over the WebSocket, saved to
disk, plus a hand-labelled ground truth (how many reps were *actually*
performed).

Replaying a clip and comparing the counted reps to the label is what turns
"the counter works on my phone" into a number we can regress against in CI.
"""

from .clips import Clip, ClipMeta, Frame, iter_clips, load_clip
from .replay import ReplayClock, ReplayResult, replay_clip
from .report import ClipScore, ExerciseScore, aggregate, markdown_table, score_clip

__all__ = [
    "Clip",
    "ClipMeta",
    "ClipScore",
    "ExerciseScore",
    "Frame",
    "ReplayClock",
    "ReplayResult",
    "aggregate",
    "iter_clips",
    "load_clip",
    "markdown_table",
    "replay_clip",
    "score_clip",
]
