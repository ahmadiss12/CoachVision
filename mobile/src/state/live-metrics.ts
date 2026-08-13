/**
 * The live workout metrics the app renders, and how they are built from the
 * WebSocket payload.
 *
 * This is the boundary where wire data becomes app state. Keeping the mapping
 * here rather than inline in the screen means it can be tested without a
 * camera, a socket, or a render.
 *
 * `app-state.jsx` holds values of this shape. It is still JavaScript, so
 * nothing enforces that yet; the type is written to match its `defaultMetrics`
 * so the two line up when it converts.
 */

import type { MetricsMessage } from '../services/ws/messages';

export type LiveMetrics = {
  /** What the big number on screen shows: reps, or seconds for a hold. */
  count: number;
  /** The counter's own tally. For holds this is completed holds, not seconds. */
  rawCount: number;
  measurementType: string | undefined;
  measurementLabel: string | undefined;
  holdDurationSec?: number | null;
  totalHoldTimeSec?: number | null;
  bestHoldSec?: number | null;
  completedHolds?: number | null;
  state: string;
  angle: number;
  feedback: string;
  progress: number;
  formName: string;
  formConfidence: number | null;
  formProbabilities?: number[] | null;
  formSource: string | null;
  confidence: number;
};

/** Session defaults used when the socket has not reported a field yet. */
export type MetricsFallback = {
  measurementType?: string | undefined;
  measurementLabel?: string | undefined;
};

/**
 * A number, or the fallback when the server sent nothing usable.
 *
 * The null check is not redundant: `Number(null)` is 0 and `Number.isFinite(0)`
 * is true, so without it a missing angle would read as 0 degrees rather than
 * falling back to 180. Unlike the original inline version this also rejects
 * NaN, which otherwise reached the UI and rendered as "NaN deg".
 */
function finiteOr(value: unknown, fallback: number): number {
  if (value === null || value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

/** Keeps "absent" distinct from zero, which `Number()` alone does not. */
function nullableNumber(value: number | null | undefined): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  return Number.isFinite(Number(value)) ? Number(value) : null;
}

/**
 * Convert a `metrics` frame into the shape the app renders.
 *
 * Hold fields are copied through as-is: they are numbers for plank and wall sit
 * and null for everything else, which is what the summary screen expects.
 */
export function metricsFromWire(
  payload: MetricsMessage,
  fallback: MetricsFallback = {},
): LiveMetrics {
  return {
    count: finiteOr(payload.count, 0),
    rawCount: finiteOr(payload.rawCount ?? payload.count, 0),
    measurementType: payload.measurementType || fallback.measurementType,
    measurementLabel: payload.measurementLabel || fallback.measurementLabel,
    holdDurationSec: payload.holdDurationSec,
    totalHoldTimeSec: payload.totalHoldTimeSec,
    bestHoldSec: payload.bestHoldSec,
    completedHolds: payload.completedHolds,
    state: payload.state,
    angle: finiteOr(payload.angle, 180),
    feedback: payload.feedback || 'Keep going.',
    progress: finiteOr(payload.progress, 0),
    formName: payload.formName || 'Correct',
    // null means the XGBoost model did not run for this frame -- a different
    // thing from a genuine score of zero, and the reason this is not a plain
    // Number() call: Number(null) is 0, and Number.isFinite(0) is true, so the
    // obvious version silently reports "no model" as "0% confidence".
    formConfidence: nullableNumber(payload.formConfidence),
    formProbabilities: payload.formProbabilities,
    formSource: payload.formSource,
    confidence: finiteOr(payload.confidence, 0),
  };
}

/**
 * Whether a new frame is worth re-rendering for.
 *
 * Pose frames arrive far faster than the eye can follow, so the screen only
 * repaints on a real change -- a new rep or a new movement phase -- and
 * otherwise throttles. Without this the UI thread does needless work at
 * roughly 15 Hz for pixels nobody can distinguish.
 */
export function shouldRenderFrame(
  next: Pick<LiveMetrics, 'rawCount' | 'count' | 'state'>,
  previous: Partial<Pick<LiveMetrics, 'rawCount' | 'count' | 'state'>> | null | undefined,
  msSinceLastRender: number,
  throttleMs: number,
): boolean {
  const previousCount = Number(previous?.rawCount ?? previous?.count ?? 0);
  const nextCount = Number(next.rawCount ?? next.count);
  if (nextCount !== previousCount) {
    return true;
  }
  if (String(next.state || '') !== String(previous?.state || '')) {
    return true;
  }
  return msSinceLastRender >= throttleMs;
}
