import { apiRequest, type Schemas } from './client';

export type FatiguePredictResponse = Schemas['FatiguePredictResponse'];
export type ExplainabilityFactorResponse = Schemas['ExplainabilityFactorResponse'];

export type ExplainabilityFactor = {
  key: string;
  label: string;
  impact: number;
  detail: string;
};

export type FatiguePrediction = {
  exerciseId: string;
  readinessScore: number;
  fatigueLevel: string;
  recommendation: string;
  factors: string[];
  generatedAt: string;
  predictionId: string | null;
  explainability: ExplainabilityFactor[];
  featureSnapshot: Record<string, unknown>;
};

function normalizeExplainability(
  item: ExplainabilityFactorResponse,
): ExplainabilityFactor {
  return {
    key: item.key,
    label: item.label,
    impact: item.impact ?? 0,
    detail: item.detail ?? '',
  };
}

/**
 * `FatiguePredictResponse` is entirely camelCase, so the snake_case fallbacks
 * the previous version carried on every field could never run. This now only
 * fills defaults.
 */
function normalizeFatiguePrediction(payload: FatiguePredictResponse): FatiguePrediction {
  return {
    exerciseId: payload.exerciseId,
    readinessScore: payload.readinessScore ?? 0,
    fatigueLevel: payload.fatigueLevel ?? 'moderate',
    recommendation: payload.recommendation ?? '',
    factors: payload.factors ?? [],
    generatedAt: payload.generatedAt,
    predictionId: payload.predictionId ?? null,
    explainability: (payload.explainability ?? []).map(normalizeExplainability),
    featureSnapshot: payload.featureSnapshot ?? {},
  };
}

export type PredictFatigueOptions = {
  exerciseId: string;
  userContext?: Record<string, unknown>;
  recentWindowDays?: number;
};

export async function predictFatigue({
  exerciseId,
  userContext = {},
  recentWindowDays = 14,
}: PredictFatigueOptions): Promise<FatiguePrediction> {
  const body: Schemas['FatiguePredictRequest'] = {
    exerciseId,
    userContext,
    recentWindowDays,
  };
  const payload = await apiRequest<FatiguePredictResponse>('/fatigue/predict', {
    method: 'POST',
    auth: true,
    body,
  });
  return normalizeFatiguePrediction(payload);
}

export async function listFatigueHistory(
  exerciseId: string,
  limit = 10,
): Promise<FatiguePrediction[]> {
  const encodedExerciseId = encodeURIComponent(exerciseId);
  const payload = await apiRequest<FatiguePredictResponse[]>(
    `/fatigue/history?exerciseId=${encodedExerciseId}&limit=${limit}`,
    { auth: true },
  );
  return Array.isArray(payload) ? payload.map(normalizeFatiguePrediction) : [];
}
