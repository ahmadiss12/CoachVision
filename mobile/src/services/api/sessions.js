import { apiRequest } from './client';

function toSessionShape(payload) {
  return {
    id: payload.id,
    exerciseId: payload.exerciseId ?? payload.exercise_id,
    difficulty: payload.difficulty,
    status: payload.status,
    targetSets: payload.targetSets ?? payload.target_sets ?? 0,
    targetReps: payload.targetReps ?? payload.target_reps ?? 0,
    totalReps: payload.totalReps ?? payload.total_reps ?? 0,
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

export async function createSession({ exerciseId, targetSets = 1, targetReps = 1, difficulty = 'beginner' }) {
  const payload = await apiRequest('/sessions', {
    method: 'POST',
    auth: true,
    body: { exerciseId, targetSets, targetReps, difficulty },
  });
  return toSessionShape(payload);
}

export async function getSession(sessionId) {
  const payload = await apiRequest(`/sessions/${sessionId}`, { auth: true });
  return toSessionShape(payload);
}

export async function startSession(sessionId) {
  const payload = await apiRequest(`/sessions/${sessionId}/start`, {
    method: 'POST',
    auth: true,
  });
  return toSessionShape(payload);
}

export async function endSession(sessionId) {
  const payload = await apiRequest(`/sessions/${sessionId}/end`, {
    method: 'POST',
    auth: true,
  });
  return toSessionShape(payload);
}

export async function getSessionFeedback(sessionId) {
  const payload = await apiRequest(`/sessions/${sessionId}/feedback`, { auth: true });
  return toFeedbackShape(payload);
}
