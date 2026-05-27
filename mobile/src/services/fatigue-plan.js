import { getExerciseMetadata, normalizeExerciseId } from '../constants/exercise-metadata';

export { normalizeExerciseId };

export function findLatestSessionForExercise(history = [], exerciseId) {
  const normalized = normalizeExerciseId(exerciseId);
  return history.find((session) => (
    normalizeExerciseId(session.exerciseId) === normalized
    || normalizeExerciseId(session.exerciseName) === normalized
  )) ?? null;
}

export function buildSuggestedRepTarget({
  prediction,
  lastSession,
  fallbackReps = 10,
  exerciseId = lastSession?.exerciseId ?? lastSession?.exerciseName,
}) {
  const readiness = prediction?.readinessScore ?? 70;
  const level = prediction?.fatigueLevel ?? 'moderate';
  const meta = getExerciseMetadata(exerciseId);
  const lastReps = Number(lastSession?.reps);
  const baseReps = Number.isFinite(lastReps) && lastReps > 0 ? lastReps : fallbackReps;
  const minimum = meta.measurementType === 'hold' ? 10 : 3;
  const unit = meta.measurementType === 'hold' ? 'seconds' : 'reps';

  if (level === 'high' || readiness < 48) {
    return {
      targetReps: Math.max(minimum, Math.round(baseReps * 0.6)),
      label: 'Recovery target',
      detail: 'Keep this light and stop early if form drops.',
      unit,
    };
  }

  if (level === 'moderate' || readiness < 72) {
    return {
      targetReps: Math.max(minimum, Math.round(baseReps * 0.85)),
      label: 'Conservative target',
      detail: 'Aim for about 10-15% less than your recent output.',
      unit,
    };
  }

  return {
    targetReps: Math.max(minimum, baseReps + (meta.measurementType === 'hold' ? 5 : baseReps >= 10 ? 2 : 1)),
    label: 'Progress target',
    detail: 'Readiness looks good, so try to match or slightly beat last time.',
    unit,
  };
}

export function fatigueLabel(level) {
  if (level === 'high') {
    return 'High fatigue';
  }
  if (level === 'low') {
    return 'Low fatigue';
  }
  return 'Moderate fatigue';
}
