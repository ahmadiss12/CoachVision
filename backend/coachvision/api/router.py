"""Top-level API router."""

from fastapi import APIRouter

from . import analytics, auth, exercises, fatigue, health, reports, sessions, users, ws_live

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, tags=["auth"])
api_router.include_router(users.router, tags=["users"])
api_router.include_router(exercises.router, tags=["exercises"])
api_router.include_router(sessions.router, tags=["sessions"])
api_router.include_router(fatigue.router, tags=["fatigue"])
api_router.include_router(analytics.router, tags=["analytics"])
api_router.include_router(reports.router, tags=["reports"])
api_router.include_router(ws_live.router, tags=["realtime"])

