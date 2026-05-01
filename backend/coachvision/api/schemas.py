"""Pydantic schemas for REST endpoints."""

from datetime import date, datetime
from uuid import UUID

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=6)
    display_name: str = Field(min_length=1, max_length=80)


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class UserMeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    display_name: str
    avatar_url: str | None = None
    date_of_birth: date | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    body_fat_percent: float | None = None
    timezone: str
    locale: str


class UpdateUserMeRequest(BaseModel):
    display_name: str | None = None
    avatar_url: str | None = None
    date_of_birth: str | None = None
    height_cm: float | None = Field(default=None, ge=80, le=260)
    weight_kg: float | None = Field(default=None, ge=20, le=400)
    body_fat_percent: float | None = Field(default=None, ge=0, le=70)
    timezone: str | None = None
    locale: str | None = None


class ExerciseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None = None
    default_difficulty: str


class CreateSessionRequest(BaseModel):
    exercise_id: str = Field(..., alias="exerciseId")
    target_sets: int = Field(1, ge=1, alias="targetSets")
    target_reps: int = Field(1, ge=1, alias="targetReps")
    difficulty: str = "intermediate"


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    exercise_id: str = Field(alias="exerciseId")
    difficulty: str
    status: str
    target_sets: int = Field(alias="targetSets")
    target_reps: int = Field(alias="targetReps")
    total_reps: int = Field(alias="totalReps")
    created_at: datetime = Field(alias="createdAt")
    started_at: datetime | None = Field(default=None, alias="startedAt")
    ended_at: datetime | None = Field(default=None, alias="endedAt")


class FatiguePredictRequest(BaseModel):
    exercise_id: str = Field(..., alias="exerciseId")
    user_context: dict = Field(
        default_factory=dict,
        alias="userContext",
        description="Optional: sleepHours, muscleSoreness (1–5), stress (1–5).",
    )
    recent_window_days: int = Field(14, ge=1, le=90, alias="recentWindowDays")


class ExplainabilityFactorResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    key: str
    label: str
    impact: int
    detail: str


class FatiguePredictResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    exercise_id: str = Field(alias="exerciseId")
    readiness_score: int = Field(alias="readinessScore")
    fatigue_level: str = Field(alias="fatigueLevel")
    recommendation: str
    factors: list[str]
    generated_at: datetime = Field(alias="generatedAt")
    prediction_id: UUID | None = Field(default=None, alias="predictionId")
    explainability: list[ExplainabilityFactorResponse] = Field(default_factory=list)
    feature_snapshot: dict = Field(default_factory=dict, alias="featureSnapshot")


class AnalyticsOverviewResponse(BaseModel):
    total_sessions: int = Field(alias="totalSessions")
    completed_sessions: int = Field(alias="completedSessions")
    avg_form_score: float | None = Field(default=None, alias="avgFormScore")


class FeedbackTopErrorItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    code: str
    label: str
    count: int
    severity: str


class FeedbackActionItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    priority: int
    title: str
    why: str
    how_to_fix: str = Field(alias="howToFix")
    cue: str


class SessionFeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    session_id: UUID = Field(alias="sessionId")
    exercise_id: str = Field(alias="exerciseId")
    overall_rating: int = Field(alias="overallRating")
    summary_text: str = Field(alias="summaryText")
    errors_count: int = Field(alias="errorsCount")
    top_errors: list[FeedbackTopErrorItem] = Field(alias="topErrors")
    error_breakdown: dict[str, dict[str, Any]] = Field(alias="errorBreakdown")
    action_items: list[FeedbackActionItem] = Field(alias="actionItems")
    confidence_overall: float | None = Field(alias="confidenceOverall")
    signals_used: dict[str, Any] = Field(alias="signalsUsed")
    version: str
    generated_at: datetime = Field(alias="generatedAt")
    updated_at: datetime = Field(alias="updatedAt")

