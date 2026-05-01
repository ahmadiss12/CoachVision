# CoachVision Backend Quickstart

This guide gets your local backend running with PostgreSQL and verifies auth in one smoke test.

## 1) Prerequisites

- Python 3.11+
- Docker Desktop (recommended for local Postgres)

## 2) Start PostgreSQL (Docker)

```powershell
docker run --name coachvision-postgres `
  -e POSTGRES_PASSWORD=postgres `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_DB=coachvision `
  -p 5432:5432 `
  -d postgres:16
```

If you already created the container before:

```powershell
docker start coachvision-postgres
```

Verify:

```powershell
docker ps
```

## 3) Configure environment

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

You can keep defaults for local development, then update `JWT_SECRET_KEY` later.

## 4) Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5) Run API

```powershell
python -m uvicorn coachvision.main:app --reload --host 127.0.0.1 --port 8001
```

Open:

- Swagger UI: `http://127.0.0.1:8001/docs`
- OpenAPI JSON: `http://127.0.0.1:8001/v1/openapi.json`

## 6) Smoke test (health -> login -> users/me)

Run in a second terminal while API is running:

```powershell
python scripts\smoke_auth.py --base-url http://127.0.0.1:8001 --email test@example.com --password test1234 --display-name "Test User"
```

Expected output ends with `Smoke test passed`.

## Notes

- App startup seeds exercise data automatically.
- Swagger `Authorize` now works with:
  - `username` = your email
  - `password` = your password
- If you see form-related errors, ensure `python-multipart` is installed via `requirements.txt`.
