# CoachVision WebSocket contract (v1.1)

See `realtime/contract.py` for `SCHEMA_VERSION` and `WsErrorCode`.

## Endpoint

`WS /v1/ws/live?token=<JWT_ACCESS_TOKEN>`

## Server messages

Every server → client JSON object includes **`schemaVersion`** (integer, currently `1`).

### Types

- `started` — `{ schemaVersion, type, sessionId }`
- `metrics` — rep count, state, angle, feedback, progress, formName, confidence
- `noPose` — no pose detected this frame
- `resetAck` — counter reset acknowledged
- `ended` — includes `summary` from dispatcher export
- `error` — `{ schemaVersion, type, sessionId?, message, code }`

### Client messages

- `start` — exerciseName, difficulty, optional targetSets/targetReps, optional sessionId (REST-linked)
- `frame` — sessionId, imageJpegBase64, optional timestampMs
- `reset` — sessionId
- `end` — sessionId

Place MediaPipe task file at: **`coachvision/ai/pose_landmarker.task`**.
