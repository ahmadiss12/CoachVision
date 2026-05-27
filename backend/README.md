# CoachVision Backend

FastAPI backend for auth, user profiles, workout sessions, realtime coaching, post-session feedback, fatigue prediction, and gamification.

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ or Docker Desktop

## Start PostgreSQL With Docker

```powershell
docker run --name coachvision-postgres `
  -e POSTGRES_PASSWORD=postgres `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_DB=coachvision `
  -p 5432:5432 `
  -d postgres:16
```

If the container already exists:

```powershell
docker start coachvision-postgres
```

## Configure And Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn coachvision.main:app --reload --host 0.0.0.0 --port 8001
```

Open:

- Swagger UI: `http://127.0.0.1:8001/docs`
- OpenAPI JSON: `http://127.0.0.1:8001/v1/openapi.json`
- Health: `http://127.0.0.1:8001/v1/health`

## Smoke Test

Run in a second terminal while the backend is running:

```powershell
python scripts\smoke_auth.py --base-url http://127.0.0.1:8001 --email test@example.com --password test1234 --display-name "Test User"
```

Expected output ends with `Smoke test passed`.

## Notes

- Startup creates missing tables and seeds the supported exercises.
- Swagger `Authorize` uses `username = email` and `password = password`.
- The optional server-side pose model path is `coachvision/ai/pose_landmarker.task`. The mobile app sends client-side landmarks, so this file is not required for normal mobile live workouts.
