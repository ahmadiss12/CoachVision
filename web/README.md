# CoachVision Web Dashboard

Next.js dashboard for **trainers** (client roster, session analytics with charts, program builder, invites) and **admins** (platform stats, user/role management). Clients use the mobile app; this dashboard rejects client logins.

It is a standalone package that talks to the same FastAPI backend as the mobile app — no server-side database access, no shared code with `mobile/`.

## Run locally

```bash
cd web
npm install
cp .env.local.example .env.local   # points at http://127.0.0.1:8001 by default
npm run dev                        # http://localhost:3001
```

Sign in with a trainer or admin account (dev admin: `admin@coachvision.test` / `Admin1234` once the backend has seeded it).

## Configuration

| Variable | Meaning |
|---|---|
| `NEXT_PUBLIC_API_ORIGIN` | Backend origin, e.g. `http://127.0.0.1:8001` or the Render URL |

## Stack

- Next.js (App Router) + React — JavaScript, no UI framework
- Hand-rolled dark design system in `app/globals.css` (matches the mobile theme)
- Dependency-free SVG charts (`components/Sparkline.js`)
- JWT auth against `/v1/auth/*` with automatic refresh (`lib/api.js`)
