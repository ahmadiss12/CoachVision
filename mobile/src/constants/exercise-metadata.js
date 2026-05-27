export const exerciseMetadata = {
  squat: {
    label: 'Squat',
    measurementType: 'reps',
    metricLabel: 'REPS',
    targetLabel: 'Target reps',
    supportsExternalLoad: true,
  },
  pushup: {
    label: 'Push-up',
    measurementType: 'reps',
    metricLabel: 'REPS',
    targetLabel: 'Target reps',
    supportsExternalLoad: false,
  },
  lunge: {
    label: 'Lunge',
    measurementType: 'reps',
    metricLabel: 'REPS',
    targetLabel: 'Target reps',
    supportsExternalLoad: true,
  },
  deadlift: {
    label: 'Deadlift',
    measurementType: 'reps',
    metricLabel: 'REPS',
    targetLabel: 'Target reps',
    supportsExternalLoad: true,
  },
  plank: {
    label: 'Plank',
    measurementType: 'hold',
    metricLabel: 'SEC',
    targetLabel: 'Target seconds',
    supportsExternalLoad: false,
  },
  bicep_curl: {
    label: 'Bicep Curl',
    measurementType: 'reps',
    metricLabel: 'REPS',
    targetLabel: 'Target reps',
    supportsExternalLoad: true,
  },
  shoulder_press: {
    label: 'Shoulder Press',
    measurementType: 'reps',
    metricLabel: 'REPS',
    targetLabel: 'Target reps',
    supportsExternalLoad: true,
  },
  situp: {
    label: 'Sit-Up',
    measurementType: 'reps',
    metricLabel: 'REPS',
    targetLabel: 'Target reps',
    supportsExternalLoad: false,
  },
  jumping_jack: {
    label: 'Jumping Jack',
    measurementType: 'reps',
    metricLabel: 'REPS',
    targetLabel: 'Target reps',
    supportsExternalLoad: false,
  },
  high_knees: {
    label: 'High Knees',
    measurementType: 'reps',
    metricLabel: 'REPS',
    targetLabel: 'Target reps',
    supportsExternalLoad: false,
  },
  mountain_climber: {
    label: 'Mountain Climber',
    measurementType: 'reps',
    metricLabel: 'REPS',
    targetLabel: 'Target reps',
    supportsExternalLoad: false,
  },
  wall_sit: {
    label: 'Wall Sit',
    measurementType: 'hold',
    metricLabel: 'SEC',
    targetLabel: 'Target seconds',
    supportsExternalLoad: false,
  },
};

export function normalizeExerciseId(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_');
}

export function getExerciseMetadata(exerciseId) {
  const normalized = normalizeExerciseId(exerciseId);
  return exerciseMetadata[normalized] ?? {
    label: normalized || 'Workout',
    measurementType: 'reps',
    metricLabel: 'REPS',
    targetLabel: 'Target reps',
    supportsExternalLoad: false,
  };
}

export function supportsExternalLoad(exerciseId) {
  return getExerciseMetadata(exerciseId).supportsExternalLoad === true;
}
