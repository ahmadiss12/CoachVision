# CoachVision

CoachVision is a full-stack fitness coaching app. It combines a FastAPI backend, an Expo React Native mobile app, realtime pose-based exercise tracking, post-session feedback, and fatigue-aware next-session planning.

## Repository Layout

```text
CoachVision-Fullstack/
  backend/                 FastAPI API, database models, AI counters, realtime WebSocket pipeline
  mobile/                  Expo React Native app
  docs/                    Project structure and product behavior notes
```

The active backend package is `backend/coachvision`. Older prototype scripts and generated training artifacts were removed so the repo points users to the maintained app path.

## Main Features

- Email/password auth with JWT access and refresh tokens.
- Profile, goals, workout history, and session feedback screens.
- Live pose tracking over WebSocket.
- Exercise-aware counting:
  - reps for dynamic exercises
  - seconds for plank and wall sit holds
- Fatigue prediction based on recent workload, form quality, sleep, soreness, stress, and external load when the exercise supports weight.

## Quick Start

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn coachvision.main:app --reload --host 0.0.0.0 --port 8001
```

API docs:

- `http://127.0.0.1:8001/docs`
- `http://127.0.0.1:8001/v1/openapi.json`

### Mobile

```powershell
cd mobile
npm install
Copy-Item .env.example .env.local
npx expo start --lan --clear
```

For a physical phone, set `EXPO_PUBLIC_API_ORIGIN` and `EXPO_PUBLIC_WS_ORIGIN` in `mobile/.env.local` to your PC LAN IP, for example:

```text
EXPO_PUBLIC_API_ORIGIN=http://192.168.0.103:8001
EXPO_PUBLIC_WS_ORIGIN=ws://192.168.0.103:8001
```

## Useful Docs

- [Project structure](docs/PROJECT_STRUCTURE.md)
- [Exercises and fatigue logic](docs/EXERCISES_AND_FATIGUE.md)
- [Counter accuracy benchmark](docs/COUNTER_BENCHMARK.md)
- [Backend details](backend/README.md)
- [Mobile details](mobile/README.md)

## Verification

Backend:

```powershell
cd backend
python -m compileall coachvision
python -m unittest coachvision.tests.test_session_feedback_generator
```

Counter accuracy (see [docs/COUNTER_BENCHMARK.md](docs/COUNTER_BENCHMARK.md)):

```powershell
cd backend
python scripts/make_synthetic_clips.py
python scripts/benchmark_counters.py --per-clip
```

Mobile:

```powershell
cd mobile
npm run lint
```
