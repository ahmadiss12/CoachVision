import { apiRequest } from './client';

function normalizeExercise(item) {
  return {
    id: item.id,
    name: item.name,
    description: item.description ?? '',
    defaultDifficulty: item.default_difficulty ?? item.defaultDifficulty ?? 'beginner',
  };
}

export async function listExercises() {
  const payload = await apiRequest('/exercises');
  return Array.isArray(payload) ? payload.map(normalizeExercise) : [];
}
