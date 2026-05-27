# CoachVision WebSocket contract (v1.1)

See `realtime/contract.py` for `SCHEMA_VERSION` and `WsErrorCode`.

## Endpoint

`WS /v1/ws/live?token=<JWT_ACCESS_TOKEN>`

## Server messages

Every server -> client JSON object includes **`schemaVersion`** (integer, currently `1`).

### Types

- `started` - `{ schemaVersion, type, sessionId }`
- `metrics` - exercise-aware `count` plus `measurementType`/`measurementLabel` (`reps`/`REPS` for dynamic exercises, `hold`/`SEC` for plank and wall sit), raw counter count in `rawCount`, state, angle, feedback, progress, formName, confidence, optional hold fields (`holdDurationSec`, `totalHoldTimeSec`, `bestHoldSec`, `completedHolds`), optional `pose` (array of `[x, y, presence]` per landmark for skeleton overlay), optional `voice` (`{ label, text }` for mobile text-to-speech), optional `serverTimingMs`
- `noPose` - no pose detected this frame, optional `serverTimingMs`
- `resetAck` - counter reset acknowledged
- `ended` - includes raw dispatcher `summary`; the richer mistake review is persisted as session feedback and returned by `GET /v1/sessions/{sessionId}/feedback`
- `error` - `{ schemaVersion, type, sessionId?, message, code }`

### Client messages

- `start` - exerciseName, difficulty, optional targetSets/targetReps, optional sessionId (REST-linked)
- `frame` - sessionId, imageJpegBase64, optional timestampMs
- `pose` - sessionId, `landmarks` array from client-side pose detection, optional timestampMs/clientInferenceMs
- `reset` - sessionId
- `end` - sessionId

Place MediaPipe task file at: **`coachvision/ai/pose_landmarker.task`**.

## Post-session mistake review

After a live workout ends, the backend converts dispatcher `rep_metrics` into a `session_feedback` row. The mobile app fetches it from:

`GET /v1/sessions/{sessionId}/feedback`

The response includes:

- `summaryText` - short coach summary.
- `topErrors` - list of wrong things detected, with count and severity.
- `errorBreakdown` - explanation and fix for each issue.
- `actionItems` - prioritized cues for the next workout.

Squat feedback is the most detailed path and can flag shallow depth, very shallow reps, limited range of motion, rushed tempo, inconsistent depth, and incomplete standing position.
