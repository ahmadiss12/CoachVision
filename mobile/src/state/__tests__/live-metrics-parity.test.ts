/**
 * Parity check against the pre-TypeScript implementation.
 *
 * `metricsFromWire` replaced an inline mapping in WorkoutLiveScreen.jsx. This
 * pins the new implementation to the old one field by field, so the conversion
 * cannot have quietly changed what the screen renders.
 *
 * The one deliberate difference is formConfidence: the original ran
 * `Number.isFinite(Number(payload.formConfidence))`, and `Number(null)` is 0,
 * so "the model did not run" was recorded as a confidence of zero. That case is
 * asserted separately below.
 */

import type { MetricsMessage } from '../../services/ws/messages';
import { metricsFromWire, type MetricsFallback } from '../live-metrics';

/** The mapping exactly as it was before the conversion. */
function originalMapping(payload: any, cfg: any) {
  return {
    count: payload.count,
    rawCount: payload.rawCount ?? payload.count,
    measurementType: payload.measurementType || cfg.measurementType,
    measurementLabel: payload.measurementLabel || cfg.metricLabel,
    holdDurationSec: payload.holdDurationSec,
    totalHoldTimeSec: payload.totalHoldTimeSec,
    bestHoldSec: payload.bestHoldSec,
    completedHolds: payload.completedHolds,
    state: payload.state,
    angle: Number(payload.angle ?? 180),
    feedback: payload.feedback || 'Keep going.',
    progress: Number(payload.progress ?? 0),
    formName: payload.formName || 'Correct',
    formConfidence: Number.isFinite(Number(payload.formConfidence))
      ? Number(payload.formConfidence)
      : null,
    formProbabilities: payload.formProbabilities || null,
    formSource: payload.formSource || null,
    confidence: Number(payload.confidence ?? 0),
  };
}

const FALLBACK: MetricsFallback = { measurementType: 'reps', measurementLabel: 'REPS' };
const LEGACY_CFG = { measurementType: 'reps', metricLabel: 'REPS' };

const FRAMES: { name: string; payload: any }[] = [
  {
    name: 'a typical rep frame',
    payload: {
      count: 7, rawCount: 7, measurementType: 'reps', measurementLabel: 'REPS',
      holdDurationSec: null, totalHoldTimeSec: null, bestHoldSec: null, completedHolds: null,
      state: 'UP', angle: 172.4, feedback: 'Good rep', progress: 0.8,
      formName: 'Correct', formConfidence: 0.91, formProbabilities: [0.9, 0.1],
      formSource: 'xgboost', confidence: 0.88,
    },
  },
  {
    name: 'a hold frame',
    payload: {
      count: 42, rawCount: 2, measurementType: 'hold', measurementLabel: 'SEC',
      holdDurationSec: 18.5, totalHoldTimeSec: 42, bestHoldSec: 24, completedHolds: 2,
      state: 'HOLDING', angle: 178, feedback: 'Hold steady', progress: 0.5,
      formName: 'Correct', formConfidence: 0.7, formProbabilities: null,
      formSource: 'rule_based', confidence: 0.9,
    },
  },
  {
    name: 'a frame with empty and missing fields',
    payload: {
      count: 0, rawCount: null, measurementType: '', measurementLabel: '',
      holdDurationSec: null, totalHoldTimeSec: null, bestHoldSec: null, completedHolds: null,
      state: 'idle', angle: null, feedback: '', progress: null,
      formName: '', formConfidence: 0.5, formProbabilities: null,
      formSource: null, confidence: null,
    },
  },
  {
    name: 'a frame with a zero count',
    payload: {
      count: 0, rawCount: 0, measurementType: 'reps', measurementLabel: 'REPS',
      holdDurationSec: null, totalHoldTimeSec: null, bestHoldSec: null, completedHolds: null,
      state: 'DOWN', angle: 90, feedback: 'Go deeper', progress: 0,
      formName: 'Shallow', formConfidence: 0, formProbabilities: [0.1, 0.9],
      formSource: 'xgboost', confidence: 0.4,
    },
  },
];

describe('metricsFromWire parity with the pre-TypeScript mapping', () => {
  it.each(FRAMES)('matches the old mapping for $name', ({ payload }) => {
    const expected = originalMapping(payload, LEGACY_CFG);
    const actual = metricsFromWire(payload as MetricsMessage, FALLBACK);
    expect(actual).toEqual(expected);
  });
});

describe('deliberate difference: NaN no longer reaches the UI', () => {
  const payload: any = {
    count: 1, rawCount: 1, measurementType: 'reps', measurementLabel: 'REPS',
    holdDurationSec: null, totalHoldTimeSec: null, bestHoldSec: null, completedHolds: null,
    state: 'UP', angle: NaN, feedback: 'ok', progress: NaN,
    formName: 'Correct', formConfidence: 0.5, formProbabilities: null,
    formSource: 'xgboost', confidence: 0.8,
  };

  it('the old mapping passed NaN straight through', () => {
    // `Number(NaN ?? 180)` is NaN, which the screen rendered as "NaN deg".
    expect(originalMapping(payload, LEGACY_CFG).angle).toBeNaN();
    expect(originalMapping(payload, LEGACY_CFG).progress).toBeNaN();
  });

  it('the new mapping falls back to a usable value', () => {
    const actual = metricsFromWire(payload as MetricsMessage, FALLBACK);
    expect(actual.angle).toBe(180);
    expect(actual.progress).toBe(0);
  });
});

describe('deliberate difference: a missing value is not zero', () => {
  const payload: any = {
    count: 1, rawCount: 1, measurementType: 'reps', measurementLabel: 'REPS',
    holdDurationSec: null, totalHoldTimeSec: null, bestHoldSec: null, completedHolds: null,
    state: 'UP', angle: 170, feedback: 'ok', progress: 0.5,
    formName: 'Correct', formConfidence: null, formProbabilities: null,
    formSource: 'rule_based', confidence: 0.8,
  };

  it('the old mapping turned "model did not run" into a score of zero', () => {
    expect(originalMapping(payload, LEGACY_CFG).formConfidence).toBe(0);
  });

  it('the new mapping keeps it null', () => {
    expect(metricsFromWire(payload as MetricsMessage, FALLBACK).formConfidence).toBeNull();
  });

  it('and still reports a genuine zero as zero', () => {
    const scored = { ...payload, formConfidence: 0 };
    expect(metricsFromWire(scored as MetricsMessage, FALLBACK).formConfidence).toBe(0);
  });
});
