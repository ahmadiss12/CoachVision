// Reads the entitlements the server sends (GET /me/entitlements). The server
// enforces every limit; these helpers only decide what the dashboard shows.

export const PLAN_ORDER = ['free', 'plus', 'pro'];

const PLAN_LABELS = { free: 'Free', plus: 'Plus', pro: 'Pro' };

// Used until the server answers, and whenever it cannot (offline, or a server
// that predates plans). Nothing is locked, so the dashboard never hides
// something the server would have allowed.
export const UNRESTRICTED = {
  plan: 'free',
  effectivePlan: 'pro',
  earlyAccess: true,
  features: null,
  limits: { exerciseIds: null, historyDays: null, maxClients: null, maxPrograms: null },
  plans: {},
  isLoaded: false,
};

export function normalizeEntitlements(payload) {
  if (!payload || typeof payload !== 'object') return UNRESTRICTED;
  return {
    plan: payload.plan ?? 'free',
    effectivePlan: payload.effectivePlan ?? payload.plan ?? 'free',
    earlyAccess: Boolean(payload.earlyAccess),
    features: payload.features ?? [],
    limits: { ...UNRESTRICTED.limits, ...(payload.limits ?? {}) },
    plans: payload.plans ?? {},
    isLoaded: true,
  };
}

/** Whether the account may use `feature` right now. */
export function hasFeature(entitlements, feature) {
  const features = entitlements?.features;
  return features == null || features.includes(feature);
}

/** Whether the account's own plan covers `exerciseId` (programs follow the trainer's plan). */
export function canUseExercise(entitlements, exerciseId) {
  const allowed = entitlements?.limits?.exerciseIds;
  return allowed == null || allowed.includes(exerciseId);
}

function lowestPlan(entitlements, test) {
  const plans = entitlements?.plans ?? {};
  const plan = PLAN_ORDER.find((name) => plans[name] && test(plans[name]));
  return plan && plan !== 'free' ? plan : null;
}

/** Lowest paid plan that includes `feature`, or null when it is free or unknown. */
export function planForFeature(entitlements, feature) {
  return lowestPlan(entitlements, (summary) => summary.features?.includes(feature));
}

/** Lowest paid plan that covers `exerciseId`, or null when it is free or unknown. */
export function planForExercise(entitlements, exerciseId) {
  return lowestPlan(entitlements, (summary) => {
    const allowed = summary.limits?.exerciseIds;
    return allowed == null || allowed.includes(exerciseId);
  });
}

/** Lowest plan that raises `limit` (e.g. 'maxClients') above the current value. */
export function planForHigherLimit(entitlements, limit) {
  const current = entitlements?.limits?.[limit];
  if (current == null) return null;
  return lowestPlan(entitlements, (summary) => {
    const value = summary.limits?.[limit];
    return value == null || value > current;
  });
}

export function planLabel(plan) {
  return PLAN_LABELS[plan] ?? 'a higher plan';
}

/** True when the server refused a request because of the account's plan. */
export function isPlanRequired(error) {
  return error?.status === 402;
}
