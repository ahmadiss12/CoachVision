# Backend Package Structure

Run the backend from `backend/`:

```powershell
python -m uvicorn coachvision.main:app --reload --host 0.0.0.0 --port 8001
```

## `coachvision/`

- `main.py`: FastAPI app setup, CORS, `/v1` router, startup bootstrap.
- `api/`: REST endpoints, WebSocket endpoint, and Pydantic schemas.
- `core/`: settings and auth helpers.
- `db/`: SQLAlchemy models, database session, seed/bootstrap logic.
- `services/`: session persistence, feedback, fatigue prediction, gamification.
- `realtime/`: live pose metric pipeline and WebSocket contract helpers.
- `ai/`: exercise counters, geometry utilities, voice cue rules.
- `repositories/`: minimal repository base for future persistence refactors.
- `infrastructure/sql/`: reference SQL schema and design docs.
- `tests/`: backend unit tests.

## Important Runtime Paths

- REST API root: `/v1`
- Live WebSocket: `/v1/ws/live?token=<access-token>`
- Optional server-side MediaPipe model: `coachvision/ai/pose_landmarker.task`

The mobile app currently sends client-side pose landmarks, so the backend can run without the optional `.task` file. If you later send JPEG frames to the backend, add the MediaPipe model at that path.

## Key Modules

- `api/auth.py`: login, register, token refresh.
- `api/sessions.py`: create, start, end, list sessions.
- `api/fatigue.py`: fatigue/readiness prediction endpoints.
- `api/ws_live.py`: realtime session socket.
- `ai/counters/dispatcher.py`: routes pose landmarks to the correct exercise counter.
- `ai/squat_form_classifier.py`: extracts the 12 training features and runs the deployed XGBoost squat-form model with per-session probability smoothing.
- `services/ws_persistence.py`: saves live workout results.
- `services/session_feedback_generator.py`: builds post-session coach feedback.
- `services/fatigue_engine.py`: transparent rule-based readiness engine.

## Tests

```powershell
python -m unittest coachvision.tests.test_session_feedback_generator
```
