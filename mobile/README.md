# CoachVision Mobile

Expo React Native frontend for CoachVision. It talks to the FastAPI backend through `/v1` REST endpoints and `/v1/ws/live` for realtime workout metrics.

## Quick Start

```powershell
npm install
Copy-Item .env.example .env.local
npx expo start --lan --clear
```

## Environment Variables

`mobile/.env.example` contains the local defaults:

```text
EXPO_PUBLIC_API_ORIGIN=http://127.0.0.1:8001
EXPO_PUBLIC_API_PREFIX=/v1
EXPO_PUBLIC_WS_ORIGIN=ws://127.0.0.1:8001
EXPO_PUBLIC_USE_MOCK_STREAM=false
```

For a physical phone, replace `127.0.0.1` with your PC LAN IP, for example:

```text
EXPO_PUBLIC_API_ORIGIN=http://192.168.0.103:8001
EXPO_PUBLIC_WS_ORIGIN=ws://192.168.0.103:8001
```

## Networking Notes

- Android emulator: use `http://10.0.2.2:8001`.
- iOS simulator: usually use `http://127.0.0.1:8001`.
- Physical iOS/Android devices: use your PC LAN IP and keep both devices on the same network.

## App Flow

- Auth: register, login, refresh token.
- Onboarding: profile and goals.
- Workout setup: choose exercise, fatigue check, target reps/seconds, optional load for weighted exercises.
- Live workout: camera pose tracking, realtime form feedback, voice cues.
- Session summary: reps or hold seconds, coach feedback, next-session fatigue outlook.
- History/profile/settings tabs.

## Quality Check

```powershell
npm run lint
```
