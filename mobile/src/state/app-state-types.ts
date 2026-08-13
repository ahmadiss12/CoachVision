/**
 * The shape of the app state store, as its consumers use it.
 *
 * `app-state.jsx` is still JavaScript and builds its context with
 * `createContext(null)`, so TypeScript infers `useAppState()` as returning
 * `never` -- no useful type reaches the screens. Declaring the shape here gives
 * consumers something real to check against, and gives `app-state` a contract
 * to implement when it converts.
 *
 * Until then this is an assertion, not a guarantee: it describes the store, it
 * does not verify it. Keep it in step with `app-state.jsx`.
 */

import type { AuthTokens } from '../services/api/client';
import type { LiveMetrics } from './live-metrics';

export type SessionConfig = {
  exerciseName?: string;
  difficulty?: string;
  targetSets?: number;
  targetReps?: number;
  measurementType?: string;
  metricLabel?: string;
  readinessContext?: Record<string, unknown> & {
    externalLoadKg?: number | null;
    bodyWeightKg?: number | null;
  };
};

export type CurrentSession = {
  sessionId?: string;
  config?: SessionConfig;
};

export type FinishWorkoutOptions = {
  endedSummary?: Record<string, unknown> | null;
};

/**
 * The slice the live workout screen uses. The store exposes more than this;
 * only what is actually consumed here is declared, so the assertion stays
 * small enough to keep honest.
 */
export type AppStateSlice = {
  authTokens: AuthTokens | null;
  currentSession: CurrentSession | null;
  updateMetrics: (metrics: LiveMetrics) => void;
  finishWorkout: (options: FinishWorkoutOptions) => Promise<unknown>;
  latestError: string | null;
  clearError: () => void;
  setLatestError: (message: string) => void;
};
