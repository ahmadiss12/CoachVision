# Voice feedback (new model)

Single-label, no-queue design to eliminate delay and backlog.

## Components

- **FeedbackPolicy** (`feedback_policy.py`): Decides at most one `FeedbackLabel` per frame from `count`, `state`, `angle`, and optional `form_name`. Squat logic: rep complete once per rep, one depth cue per rep in DOWN, one state cue per transition. Other exercises return `NONE` (extend as needed).

- **VoiceCoach** (`voice_coach.py`): Single-slot playback. `say(label)` overwrites any pending request (latest wins). Uses pre-generated audio (edge-tts at startup); playback is non-blocking and can be stopped immediately. `stop()` does not wait for playback to finish.

- **phrases** (`phrases.py`): Maps exercise + difficulty + `FeedbackLabel` to phrase text (squat only by default).

## Flow

1. At startup: `FeedbackPolicy(exercise, difficulty)`, `VoiceCoach(...)`, `coach.pregenerate()`.
2. Each frame: `label = policy.decide(count, state, angle, form_name)`. If `label != NONE` and (label changed or cooldown ≥ 1s): `coach.say(label)`.
3. On exit: `coach.stop()` — no waiting.

## Result

- No queue, no backlog; at most one phrase in flight.
- Exit stops voice immediately (no trailing speech after video ends).
- One depth cue per rep, one rep-complete per rep, one instruction per state transition.
