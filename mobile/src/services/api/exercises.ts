import { apiRequest, type Schemas } from './client';

export type ExerciseResponse = Schemas['ExerciseResponse'];

export type Exercise = {
  id: string;
  name: string;
  description: string;
  defaultDifficulty: string;
};

/**
 * `default_difficulty` is snake_case on the wire -- one of the eighteen
 * properties that break the API's otherwise camelCase convention. The previous
 * version also read `item.defaultDifficulty`, which the schema shows never
 * exists.
 */
function normalizeExercise(item: ExerciseResponse): Exercise {
  return {
    id: item.id,
    name: item.name,
    description: item.description ?? '',
    defaultDifficulty: item.default_difficulty ?? 'beginner',
  };
}

export async function listExercises(): Promise<Exercise[]> {
  const payload = await apiRequest<ExerciseResponse[]>('/exercises');
  return Array.isArray(payload) ? payload.map(normalizeExercise) : [];
}
