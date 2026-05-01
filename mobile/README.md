# CoachVision Mobile

React Native (Expo) frontend for CoachVision, integrated with the FastAPI backend (`/v1` REST + `/v1/ws/live` realtime channel).

## Quick start

1. Install dependencies:

   ```bash
   npm install
   ```

2. Configure backend endpoints (PowerShell):

   ```powershell
   $env:EXPO_PUBLIC_API_ORIGIN="http://127.0.0.1:8000"
   $env:EXPO_PUBLIC_API_PREFIX="/v1"
   $env:EXPO_PUBLIC_WS_ORIGIN="ws://127.0.0.1:8000"
   ```

3. Start frontend:

   ```bash
   npx expo start
   ```

## Environment variables

- `EXPO_PUBLIC_API_ORIGIN`: backend origin (example `http://127.0.0.1:8000`)
- `EXPO_PUBLIC_API_PREFIX`: API prefix (default `/v1`)
- `EXPO_PUBLIC_WS_ORIGIN`: websocket origin (example `ws://127.0.0.1:8000`)
- `EXPO_PUBLIC_USE_MOCK_STREAM`: set to `true` to force old mock workout stream fallback

If `EXPO_PUBLIC_WS_ORIGIN` is missing, the app derives it from `EXPO_PUBLIC_API_ORIGIN` (`http` -> `ws`).

## Device networking notes

- Android emulator can use `http://10.0.2.2:8000` for a backend running on your PC.
- iOS simulator can usually use `http://127.0.0.1:8000`.
- Physical devices must use your machine LAN IP (example `http://192.168.x.x:8000`) and same network.

## Integrated backend flow

- Auth: `POST /v1/auth/register`, `POST /v1/auth/login`, `POST /v1/auth/refresh`
- Profile: `GET/PATCH /v1/users/me`
- Sessions: `POST /v1/sessions`, `POST /v1/sessions/{id}/start`, `POST /v1/sessions/{id}/end`
- Feedback: `GET /v1/sessions/{id}/feedback`
- Realtime: `WS /v1/ws/live?token=<accessJWT>`

## Verification checklist

1. Register a user from the mobile app.
2. Log in and complete onboarding screens.
3. Start a workout from setup and verify live metrics update.
4. End session and confirm summary screen shows backend feedback.
5. Open history and confirm completed session appears.

## Quality checks

Run lint:

```bash
npm run lint
```
