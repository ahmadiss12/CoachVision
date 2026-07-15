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
    role: str = Field(default="client", pattern="^(client|trainer)$")


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
    role: str
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


class AdminUserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    email: str
    display_name: str = Field(alias="displayName")
    role: str
    created_at: datetime = Field(alias="createdAt")


class UpdateRoleRequest(BaseModel):
    role: str = Field(pattern="^(client|trainer|admin)$")


class AdminStatsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total_users: int = Field(alias="totalUsers")
    clients: int
    trainers: int
    admins: int
    completed_sessions: int = Field(alias="completedSessions")
    sessions_last_7d: int = Field(alias="sessionsLast7d")
    active_coaching_links: int = Field(alias="activeCoachingLinks")
    pending_invites: int = Field(alias="pendingInvites")


class InviteCreateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    email: str | None = Field(
        default=None, description="Optional: restrict the invite to this email."
    )
    expires_in_days: int = Field(7, ge=1, le=30, alias="expiresInDays")


class InviteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    token: str
    email: str | None = None
    status: str
    expires_at: datetime = Field(alias="expiresAt")
    created_at: datetime = Field(alias="createdAt")
    accepted_at: datetime | None = Field(default=None, alias="acceptedAt")


class InviteAcceptRequest(BaseModel):
    token: str = Field(min_length=1)


class InviteAcceptResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    trainer_id: UUID = Field(alias="trainerId")
    trainer_name: str = Field(alias="trainerName")
    status: str


class TrainerClientResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    client_id: UUID = Field(alias="clientId")
    display_name: str = Field(alias="displayName")
    email: str
    status: str
    linked_at: datetime = Field(alias="linkedAt")


class ProgramExerciseInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    exercise_id: str = Field(alias="exerciseId")
    target_sets: int = Field(1, ge=1, le=20, alias="targetSets")
    target_reps: int = Field(1, ge=1, le=500, alias="targetReps")
    rest_seconds: int = Field(60, ge=0, le=600, alias="restSeconds")
    external_load_kg: float | None = Field(default=None, ge=0, le=300, alias="externalLoadKg")
    notes: str | None = Field(default=None, max_length=300)


class ProgramDayInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    day_index: int = Field(ge=1, le=28, alias="dayIndex")
    notes: str | None = None
    exercises: list[ProgramExerciseInput] = Field(default_factory=list)


class ProgramCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = None
    weeks: int = Field(1, ge=1, le=4)
    days: list[ProgramDayInput] = Field(default_factory=list)


class ProgramExerciseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    exercise_id: str = Field(alias="exerciseId")
    position: int
    target_sets: int = Field(alias="targetSets")
    target_reps: int = Field(alias="targetReps")
    rest_seconds: int = Field(alias="restSeconds")
    external_load_kg: float | None = Field(default=None, alias="externalLoadKg")
    notes: str | None = None


class ProgramDayResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    day_index: int = Field(alias="dayIndex")
    notes: str | None = None
    exercises: list[ProgramExerciseResponse] = Field(default_factory=list)


class ProgramResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    name: str
    description: str | None = None
    weeks: int
    created_at: datetime = Field(alias="createdAt")
    days: list[ProgramDayResponse] = Field(default_factory=list)


class AssignProgramRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    client_id: UUID = Field(alias="clientId")
    start_date: str | None = Field(default=None, alias="startDate", description="YYYY-MM-DD; defaults to today")


class AssignmentResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: UUID
    program_id: UUID = Field(alias="programId")
    program_name: str = Field(alias="programName")
    client_id: UUID = Field(alias="clientId")
    start_date: date = Field(alias="startDate")
    status: str
    created_at: datetime = Field(alias="createdAt")
    days_elapsed: int | None = Field(default=None, alias="daysElapsed")
    sessions_completed: int = Field(default=0, alias="sessionsCompleted")


class PlanExerciseResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    exercise_id: str = Field(alias="exerciseId")
    exercise_name: str = Field(alias="exerciseName")
    prescribed_sets: int = Field(alias="prescribedSets")
    prescribed_reps: int = Field(alias="prescribedReps")
    adjusted_sets: int = Field(alias="adjustedSets")
    adjusted_reps: int = Field(alias="adjustedReps")
    rest_seconds: int = Field(alias="restSeconds")
    external_load_kg: float | None = Field(default=None, alias="externalLoadKg")
    notes: str | None = None
    readiness_score: int = Field(alias="readinessScore")
    fatigue_level: str = Field(alias="fatigueLevel")
    adjustment_note: str | None = Field(default=None, alias="adjustmentNote")


class TodayPlanResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    has_plan: bool = Field(alias="hasPlan")
    rest_day: bool = Field(default=False, alias="restDay")
    program_name: str | None = Field(default=None, alias="programName")
    trainer_name: str | None = Field(default=None, alias="trainerName")
    assignment_id: UUID | None = Field(default=None, alias="assignmentId")
    day_index: int | None = Field(default=None, alias="dayIndex")
    cycle_length: int | None = Field(default=None, alias="cycleLength")
    day_notes: str | None = Field(default=None, alias="dayNotes")
    message: str | None = None
    exercises: list[PlanExerciseResponse] = Field(default_factory=list)


class BodyMetricCreateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date: str
    weight_kg: float = Field(..., ge=20, le=400, alias="weightKg")
    body_fat_percent: float = Field(..., ge=0, le=70, alias="bodyFatPercent")


class BodyMetricResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    entry_date: date = Field(alias="date")
    weight_kg: float = Field(alias="weightKg")
    body_fat_percent: float = Field(alias="bodyFatPercent")
    created_at: datetime = Field(alias="createdAt")


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
    external_load_kg: float | None = Field(default=None, ge=0, le=300, alias="externalLoadKg")
    body_weight_kg: float | None = Field(default=None, ge=20, le=400, alias="bodyWeightKg")
    assignment_id: UUID | None = Field(default=None, alias="assignmentId")


class EndSessionRequest(BaseModel):
    total_reps: int | None = Field(default=None, ge=0, alias="totalReps")
    avg_form_score: float | None = Field(default=None, ge=0, le=100, alias="avgFormScore")
    external_load_kg: float | None = Field(default=None, ge=0, le=300, alias="externalLoadKg")
    body_weight_kg: float | None = Field(default=None, ge=20, le=400, alias="bodyWeightKg")


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    exercise_id: str = Field(alias="exerciseId")
    difficulty: str
    status: str
    target_sets: int = Field(alias="targetSets")
    target_reps: int = Field(alias="targetReps")
    external_load_kg: float | None = Field(default=None, alias="externalLoadKg")
    body_weight_kg: float | None = Field(default=None, alias="bodyWeightKg")
    total_reps: int = Field(alias="totalReps")
    avg_form_score: float | None = Field(default=None, alias="avgFormScore")
    duration_seconds: int | None = Field(default=None, alias="durationSeconds")
    created_at: datetime = Field(alias="createdAt")
    started_at: datetime | None = Field(default=None, alias="startedAt")
    ended_at: datetime | None = Field(default=None, alias="endedAt")


class FatiguePredictRequest(BaseModel):
    exercise_id: str = Field(..., alias="exerciseId")
    user_context: dict = Field(
        default_factory=dict,
        alias="userContext",
        description="Optional: sleepHours, muscleSoreness (1-5), stress (1-5), externalLoadKg, bodyWeightKg.",
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


class ReportUserResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: UUID
    email: str
    display_name: str = Field(alias="displayName")


class DailyReportSummaryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date: date
    timezone: str
    total_sessions: int = Field(alias="totalSessions")
    total_reps: int = Field(alias="totalReps")
    total_duration_seconds: int = Field(alias="totalDurationSeconds")
    avg_form_score: float | None = Field(default=None, alias="avgFormScore")
    total_external_load_volume_kg: float | None = Field(
        default=None,
        alias="totalExternalLoadVolumeKg",
        description="Sum of externalLoadKg * reps for weighted rep-based workouts.",
    )


class ReportRepStatsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    rep_count: int = Field(alias="repCount")
    avg_range_of_motion: float | None = Field(default=None, alias="avgRangeOfMotion")
    avg_rep_duration_ms: float | None = Field(default=None, alias="avgRepDurationMs")
    avg_form_score: float | None = Field(default=None, alias="avgFormScore")
    depth_quality_counts: dict[str, int] = Field(default_factory=dict, alias="depthQualityCounts")


class DailyReportSessionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: UUID = Field(alias="sessionId")
    exercise_id: str = Field(alias="exerciseId")
    exercise_name: str = Field(alias="exerciseName")
    difficulty: str
    target_sets: int = Field(alias="targetSets")
    target_reps: int = Field(alias="targetReps")
    total_reps: int = Field(alias="totalReps")
    avg_form_score: float | None = Field(default=None, alias="avgFormScore")
    duration_seconds: int | None = Field(default=None, alias="durationSeconds")
    started_at: datetime | None = Field(default=None, alias="startedAt")
    ended_at: datetime | None = Field(default=None, alias="endedAt")
    external_load_kg: float | None = Field(default=None, alias="externalLoadKg")
    body_weight_kg: float | None = Field(default=None, alias="bodyWeightKg")
    uses_external_load: bool = Field(alias="usesExternalLoad")
    estimated_volume_load_kg: float | None = Field(default=None, alias="estimatedVolumeLoadKg")
    feedback: SessionFeedbackResponse | None = None
    rep_stats: ReportRepStatsResponse = Field(alias="repStats")


class DailyReportResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    generated_at: datetime = Field(alias="generatedAt")
    user: ReportUserResponse
    summary: DailyReportSummaryResponse
    sessions: list[DailyReportSessionResponse]

