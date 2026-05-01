# CoachVision package layout

Monolithic backend + AI code under one import root: **`coachvision`**. Run from repo `fyp1/`:

```bash
uvicorn coachvision.main:app --reload --host 0.0.0.0 --port 8000
```

Ensure `PYTHONPATH` includes `fyp1` (or run from `fyp1`).

---

## `coachvision/` (root)

- **`__init__.py`** — package version.
- **`main.py`** — FastAPI app factory, mounts `/v1` router, runs DB bootstrap on startup.

---

## `coachvision/api/`

HTTP + WebSocket route modules and shared API types.

- **`router.py`** — includes all sub-routers under `/v1`.
- **`auth.py`**, **`users.py`**, **`exercises.py`**, **`sessions.py`**, **`fatigue.py`**, **`analytics.py`** — REST endpoints.
- **`ws_live.py`** — live coaching WebSocket (`/v1/ws/live`); persists sessions; adds `schemaVersion` on all outbound messages.
- **`deps.py`** — JWT user dependency for REST.
- **`schemas.py`** — Pydantic request/response models.

---

## `coachvision/core/`

Cross-cutting app configuration and security.

- **`config.py`** — settings (`DATABASE_URL`, JWT, API prefix).
- **`security.py`** — password hashing, JWT encode/decode.

---

## `coachvision/db/`

SQLAlchemy persistence.

- **`base.py`** — declarative base.
- **`models.py`** — users, sessions, `session_feedback`, rep_events, fatigue, gamification, calibration, etc.
- **`session.py`** — engine + `SessionLocal` + `get_db`.
- **`bootstrap.py`** — `create_all` + seed `exercises`.

---

## `coachvision/services/`

Business logic not tied to HTTP.

- **`ws_persistence.py`** — session rows + `rep_events` + `session_feedback` on live workout end.
- **`session_feedback_generator.py`** — rule-based recap from exporter `rep_metrics` / summary.
- **`fatigue_*`** — rolling features, rule engine, daily rollup, calibration, post-session hook.
- **`gamification_service.py`** — streaks, XP, achievements after completed session.
- **`fatigue_post_session.py`** — orchestrates fatigue + gamification after completion.
- **`session_service.py`**, **`fatigue_service.py`** — placeholders for future refactors.

---

## `coachvision/realtime/`

Live inference path for WebSocket frames.

- **`pipeline.py`** — JPEG decode → MediaPipe pose → `ExerciseDispatcher` metrics.
- **`contract.py`** — `SCHEMA_VERSION`, `WsErrorCode`.
- **`connection_manager.py`**, **`schemas.py`** — helpers / optional alternate message shapes.

Model file path: **`coachvision/ai/pose_landmarker.task`** (not committed; download separately).

---

## `coachvision/ai/`

Exercise intelligence: counters, geometry, voice policy, optional form analyzer.

- **`counters/`** — per-exercise FSM + rep metrics; **`dispatcher.py`** is the entry used by the realtime pipeline.
- **`utils/`** — geometry + OneEuro filtering.
- **`voice/`** — coaching phrases / policy (used by desktop demo; WS can stay text-only).
- **`squat_form_analyzer.py`** — optional classifier hook for squats.

---

## `coachvision/repositories/`

Thin repository base for future “clean architecture” extraction (currently minimal).

---

## `coachvision/infrastructure/`

Non-Python deployment assets.

- **`sql/database_schema_v1.sql`** — reference Postgres DDL (optional if you use SQLAlchemy `create_all` only).

---

## `coachvision/tests/`

- **`test_session_feedback_generator.py`** — unit tests for recap rules and `rep_events` / REST export helpers (`python -m unittest coachvision.tests.test_session_feedback_generator` from `fyp1/` with `PYTHONPATH` set).

---

## Cleanup note

Ignore or delete stray **`__pycache__`** and **`.ipynb_checkpoints/`** under `ai/` when committing.
