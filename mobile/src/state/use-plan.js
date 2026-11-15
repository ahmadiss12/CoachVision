import { useMemo } from 'react';
import {
  canUseExercise,
  hasFeature,
  planForExercise,
  planForFeature,
  planForHigherLimit,
} from '../services/plan-access';
import { useAppState } from './app-state';

/** The signed-in account's plan, with the questions screens ask about it. */
export function usePlan() {
  const { entitlements } = useAppState();
  return useMemo(() => ({
    entitlements,
    plan: entitlements.plan,
    limits: entitlements.limits,
    // Paid features are open to everyone for now; screens label them instead of locking.
    earlyAccess: entitlements.isLoaded && entitlements.earlyAccess,
    can: (feature) => hasFeature(entitlements, feature),
    canUseExercise: (exerciseId) => canUseExercise(entitlements, exerciseId),
    planForFeature: (feature) => planForFeature(entitlements, feature),
    planForExercise: (exerciseId) => planForExercise(entitlements, exerciseId),
    planForHigherLimit: (limit) => planForHigherLimit(entitlements, limit),
  }), [entitlements]);
}
