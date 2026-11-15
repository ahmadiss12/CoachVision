// Reads the entitlements the server sends (GET /me/entitlements). The server is
// what enforces every limit; these helpers only decide what the app shows.

export const PLAN_ORDER = ['free', 'plus', 'pro'];

export const PLAN_LABELS = {
  free: 'Free',
  plus: 'Plus',
  pro: 'Pro',
};

// Used until the server answers, and whenever it cannot (offline, or a server
// that predates plans). Nothing is locked, so the app never hides a feature
// the server would have allowed.
export const UNRESTRICTED_ENTITLEMENTS = {
  plan: 'free',
  effectivePlan: 'pro',
  earlyAccess: true,
  features: null,
  limits: { exerciseIds: null, historyDays: null, maxClients: null, maxPrograms: null },
  plans: {},
  sponsoredExerciseIds: [],
  isLoaded: false,
};

function normalizeLimits(limits = {}) {
  return {
    exerciseIds: limits.exerciseIds ?? limits.exercise_ids ?? null,
    historyDays: limits.historyDays ?? limits.history_days ?? null,
    maxClients: limits.maxClients ?? limits.max_clients ?? null,
    maxPrograms: limits.maxPrograms ?? limits.max_programs ?? null,
  };
}

export function normalizeEntitlements(payload) {
  if (!payload || typeof payload !== 'object') {
    return UNRESTRICTED_ENTITLEMENTS;
  }
  const plans = {};
  Object.entries(payload.plans ?? {}).forEach(([plan, summary]) => {
    plans[plan] = {
      features: summary?.features ?? [],
      limits: normalizeLimits(summary?.limits),
    };
  });
  return {
    plan: payload.plan ?? 'free',
    effectivePlan: payload.effectivePlan ?? payload.effective_plan ?? payload.plan ?? 'free',
    earlyAccess: Boolean(payload.earlyAccess ?? payload.early_access),
    features: payload.features ?? [],
    limits: normalizeLimits(payload.limits),
    plans,
    sponsoredExerciseIds: payload.sponsoredExerciseIds ?? payload.sponsored_exercise_ids ?? [],
    isLoaded: true,
  };
}

/** Whether the account may use `feature` right now. */
export function hasFeature(entitlements, feature) {
  const features = entitlements?.features;
  return features == null || features.includes(feature);
}

/** Whether the account may start `exerciseId` right now, on its own plan or a trainer's. */
export function canUseExercise(entitlements, exerciseId) {
  const allowed = entitlements?.limits?.exerciseIds;
  if (allowed == null || allowed.includes(exerciseId)) {
    return true;
  }
  return (entitlements?.sponsoredExerciseIds ?? []).includes(exerciseId);
}

/** Lowest plan that includes `feature`, or null when it is free or unknown. */
export function planForFeature(entitlements, feature) {
  const plans = entitlements?.plans ?? {};
  const plan = PLAN_ORDER.find((name) => plans[name]?.features?.includes(feature));
  return plan && plan !== 'free' ? plan : null;
}

/** Lowest plan that includes `exerciseId`, or null when it is free or unknown. */
export function planForExercise(entitlements, exerciseId) {
  const plans = entitlements?.plans ?? {};
  const plan = PLAN_ORDER.find((name) => {
    if (!plans[name]) {
      return false;
    }
    const allowed = plans[name].limits.exerciseIds;
    return allowed == null || allowed.includes(exerciseId);
  });
  return plan && plan !== 'free' ? plan : null;
}

/** Lowest plan above the current one that raises `limit` (e.g. 'maxClients'). */
export function planForHigherLimit(entitlements, limit) {
  const plans = entitlements?.plans ?? {};
  const current = entitlements?.limits?.[limit];
  if (current == null) {
    return null;
  }
  return PLAN_ORDER.find((name) => {
    const value = plans[name]?.limits?.[limit];
    return plans[name] && (value == null || value > current);
  }) ?? null;
}

export function planLabel(plan) {
  return PLAN_LABELS[plan] ?? 'a higher plan';
}

/** True when the server refused a request because of the account's plan. */
export function isPlanRequiredError(error) {
  return error?.status === 402;
}
