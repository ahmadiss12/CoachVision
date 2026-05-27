# Project Structure

This repo has two maintained applications and a small docs folder.

```text
backend/
  coachvision/
    ai/             Exercise counters, geometry helpers, voice cue policy
    api/            REST and WebSocket routes
    core/           Settings and auth helpers
    db/             SQLAlchemy models, session factory, bootstrap seed data
    realtime/       Pose processing and live metric payloads
    services/       Fatigue, feedback, gamification, persistence logic
    tests/          Backend unit tests
  scripts/          Small local smoke-test helpers
  .env.example      Local backend environment template
  requirements.txt  Backend dependencies

mobile/
  app/              Expo Router routes
  images/           Exercise images used by workout setup
  src/
    components/     Shared UI components
    constants/      Exercise metadata and pose constants
    screens/        App screens
    services/       REST, WebSocket, fatigue planning, mock stream helpers
    state/          App-level state provider
    theme/          Theme tokens
  .env.example      Local mobile environment template
  package.json      Expo scripts and dependencies

docs/
  PROJECT_STRUCTURE.md
  EXERCISES_AND_FATIGUE.md
```

## Runtime Flow

1. The mobile app creates and starts a session through the REST API.
2. The live workout screen opens `/v1/ws/live` with the access token.
3. Mobile MediaPipe sends pose landmarks to the backend.
4. `coachvision.realtime.pipeline` sends landmarks to `ExerciseDispatcher`.
5. The dispatcher returns either rep metrics or hold-time metrics.
6. When the session ends, `ws_persistence` saves the session and rep/hold events.
7. `session_feedback_generator` builds the coach summary.
8. `fatigue_engine` predicts next-session readiness from recent workload and self-reported recovery.

## Repo Hygiene

Generated logs, virtual environments, Expo caches, local env files, downloaded datasets, archives, and large local ML artifacts are ignored by Git. Commit source code, docs, environment examples, and small static assets only.
