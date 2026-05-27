import { apiRequest } from './client';

function normalizeExplainability(item) {
  return {
    key: item.key,
    label: item.label,
    impact: item.impact ?? 0,
    detail: item.detail ?? '',
  };
}

function normalizeFatiguePrediction(payload) {
  return {
    exerciseId: payload.exerciseId ?? payload.exercise_id,
    readinessScore: payload.readinessScore ?? payload.readiness_score ?? 0,
    fatigueLevel: payload.fatigueLevel ?? payload.fatigue_level ?? 'moderate',
    recommendation: payload.recommendation ?? '',
    factors: payload.factors ?? [],
    generatedAt: payload.generatedAt ?? payload.generated_at,
    predictionId: payload.predictionId ?? payload.prediction_id ?? null,
    explainability: (payload.explainability ?? []).map(normalizeExplainability),
    featureSnapshot: payload.featureSnapshot ?? payload.feature_snapshot ?? {},
  };
}

export async function predictFatigue({
  exerciseId,
  userContext = {},
  recentWindowDays = 14,
}) {
  const payload = await apiRequest('/fatigue/predict', {
    method: 'POST',
    auth: true,
    body: {
      exerciseId,
      userContext,
      recentWindowDays,
    },
  });
  return normalizeFatiguePrediction(payload);
}

export async function listFatigueHistory(exerciseId, limit = 10) {
  const encodedExerciseId = encodeURIComponent(exerciseId);
  const payload = await apiRequest(`/fatigue/history?exerciseId=${encodedExerciseId}&limit=${limit}`, {
    auth: true,
  });
  return Array.isArray(payload) ? payload.map(normalizeFatiguePrediction) : [];
}
