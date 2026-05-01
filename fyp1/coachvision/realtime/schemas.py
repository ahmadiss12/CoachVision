"""Realtime WebSocket message schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class BaseRealtimeMessage(BaseModel):
    type: str
    ts: int = Field(..., description="Unix epoch in milliseconds")


class FrameLandmarksMessage(BaseRealtimeMessage):
    type: Literal["frame_landmarks"] = "frame_landmarks"
    exercise_id: str = Field(..., alias="exerciseId")
    landmarks: list[dict[str, Any]]
    device_meta: dict[str, Any] = Field(default_factory=dict, alias="deviceMeta")


class AnalysisUpdateMessage(BaseRealtimeMessage):
    type: Literal["analysis_update"] = "analysis_update"
    rep_count: int = Field(..., alias="repCount")
    form_score: float = Field(..., alias="formScore")
    coaching_text: str = Field(..., alias="coachingText")
    issues: list[str] = Field(default_factory=list)

