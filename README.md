# CoachVision Fullstack Monorepo

This repository contains both projects:

- `backend/` - FastAPI backend (copied from `CoachVision/CoachVision/fyp1`)
- `mobile/` - Expo React Native frontend (copied from `FYP/CoachVision-Mobile`)

## Quick start

### Backend

```powershell
cd backend
python -m uvicorn coachvision.main:app --reload --host 0.0.0.0 --port 8001
```

### Mobile

```powershell
cd mobile
npm install
$env:EXPO_PUBLIC_API_ORIGIN="http://<YOUR_PC_IP>:8001"
$env:EXPO_PUBLIC_API_PREFIX="/v1"
$env:EXPO_PUBLIC_WS_ORIGIN="ws://<YOUR_PC_IP>:8001"
npx expo start --lan --clear
```
