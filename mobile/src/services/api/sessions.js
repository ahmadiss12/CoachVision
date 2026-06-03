import { apiRequest } from './client';

function toSessionShape(payload) {
  return {
    id: payload.id,
    exerciseId: payload.exerciseId ?? payload.exercise_id,
    difficulty: payload.difficulty,
    status: payload.status,
    targetSets: payload.targetSets ?? payload.target_sets ?? 0,
    targetReps: payload.targetReps ?? payload.target_reps ?? 0,
    externalLoadKg: payload.externalLoadKg ?? payload.external_load_kg ?? null,
    bodyWeightKg: payload.bodyWeightKg ?? payload.body_weight_kg ?? null,
    totalReps: payload.totalReps ?? payload.total_reps ?? 0,
    avgFormScore: payload.avgFormScore ?? payload.avg_form_score ?? null,
    durationSeconds: payload.durationSeconds ?? payload.duration_seconds ?? null,
    createdAt: payload.createdAt ?? payload.created_at,
    startedAt: payload.startedAt ?? payload.started_at ?? null,
    endedAt: payload.endedAt ?? payload.ended_at ?? null,
  };
}

function toFeedbackShape(payload) {
  return {
    id: payload.id,
    sessionId: payload.sessionId ?? payload.session_id,
    exerciseId: payload.exerciseId ?? payload.exercise_id,
    overallRating: payload.overallRating ?? payload.overall_rating ?? 0,
    summaryText: payload.summaryText ?? payload.summary_text ?? '',
    errorsCount: payload.errorsCount ?? payload.errors_count ?? 0,
    topErrors: payload.topErrors ?? payload.top_errors ?? [],
    errorBreakdown: payload.errorBreakdown ?? payload.error_breakdown ?? {},
    actionItems: payload.actionItems ?? payload.action_items ?? [],
    confidenceOverall: payload.confidenceOverall ?? payload.confidence_overall ?? null,
    signalsUsed: payload.signalsUsed ?? payload.signals_used ?? {},
    version: payload.version,
    generatedAt: payload.generatedAt ?? payload.generated_at,
    updatedAt: payload.updatedAt ?? payload.updated_at,
  };
}

export async function createSession({
  exerciseId,
  targetSets = 1,
  targetReps = 1,
  difficulty = 'beginner',
  externalLoadKg = null,
  bodyWeightKg = null,
}) {
  const payload = await apiRequest('/sessions', {
    method: 'POST',
    auth: true,
    body: { exerciseId, targetSets, targetReps, difficulty, externalLoadKg, bodyWeightKg },
  });
  return toSessionShape(payload);
}

export async function getSession(sessionId) {
  const payload = await apiRequest(`/sessions/${sessionId}`, { auth: true });
  return toSessionShape(payload);
}

export async function listSessions() {
  const payload = await apiRequest('/sessions', { auth: true });
  return Array.isArray(payload) ? payload.map(toSessionShape) : [];
}

export async function startSession(sessionId) {
  const payload = await apiRequest(`/sessions/${sessionId}/start`, {
    method: 'POST',
    auth: true,
  });
  return toSessionShape(payload);
}

export async function endSession(sessionId, summary = {}) {
  const payload = await apiRequest(`/sessions/${sessionId}/end`, {
    method: 'POST',
    auth: true,
    body: {
      totalReps: summary.totalReps ?? summary.total_reps ?? 0,
      avgFormScore: summary.avgFormScore ?? summary.avg_form_score ?? null,
      externalLoadKg: summary.externalLoadKg ?? summary.external_load_kg ?? null,
      bodyWeightKg: summary.bodyWeightKg ?? summary.body_weight_kg ?? null,
    },
  });
  return toSessionShape(payload);
}

export async function getSessionFeedback(sessionId) {
  const payload = await apiRequest(`/sessions/${sessionId}/feedback`, { auth: true });
  return toFeedbackShape(payload);
}
