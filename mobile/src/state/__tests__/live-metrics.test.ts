import type {
  HoldMetricsMessage,
  RepMetricsMessage,
} from '../../services/ws/messages';
import { metricsFromWire, shouldRenderFrame } from '../live-metrics';

function repFrame(overrides: Partial<RepMetricsMessage> = {}): RepMetricsMessage {
  return {
    schemaVersion: 1,
    type: 'metrics',
    sessionId: 'session-1',
    count: 7,
    rawCount: 7,
    measurementType: 'reps',
    measurementLabel: 'REPS',
    holdDurationSec: null,
    totalHoldTimeSec: null,
    bestHoldSec: null,
    completedHolds: null,
    state: 'UP',
    angle: 172.4,
    feedback: 'Good rep',
    progress: 0.8,
    formName: 'Correct',
    formConfidence: 0.91,
    formProbabilities: [0.91, 0.05, 0.04],
    formSource: 'xgboost',
    confidence: 0.88,
    pose: null,
    voice: { label: 'rep_complete_7', text: 'Good rep' },
    serverTimingMs: { total: 12.4 },
    ...overrides,
  };
}

function holdFrame(overrides: Partial<HoldMetricsMessage> = {}): HoldMetricsMessage {
  return {
    ...repFrame(),
    measurementType: 'hold',
    measurementLabel: 'SEC',
    count: 42,
    rawCount: 2,
    holdDurationSec: 18.5,
    totalHoldTimeSec: 42.0,
    bestHoldSec: 24.0,
    completedHolds: 2,
    state: 'HOLDING',
    ...overrides,
  };
}

describe('metricsFromWire', () => {
  it('carries a rep frame through unchanged', () => {
    const result = metricsFromWire(repFrame());
    expect(result.count).toBe(7);
    expect(result.rawCount).toBe(7);
    expect(result.measurementLabel).toBe('REPS');
    expect(result.formConfidence).toBeCloseTo(0.91);
  });

  it('keeps hold fields for plank and wall sit', () => {
    const result = metricsFromWire(holdFrame());
    expect(result.measurementType).toBe('hold');
    expect(result.count).toBe(42);
    expect(result.bestHoldSec).toBe(24.0);
    expect(result.completedHolds).toBe(2);
  });

  it('leaves hold fields null on a rep exercise', () => {
    const result = metricsFromWire(repFrame());
    expect(result.holdDurationSec).toBeNull();
    expect(result.bestHoldSec).toBeNull();
  });

  it('falls back to the session config when the server omits the measurement', () => {
    const result = metricsFromWire(
      repFrame({ measurementType: '' as 'reps', measurementLabel: '' as 'REPS' }),
      { measurementType: 'hold', measurementLabel: 'SEC' },
    );
    expect(result.measurementType).toBe('hold');
    expect(result.measurementLabel).toBe('SEC');
  });

  it('distinguishes "model did not run" from a low score', () => {
    // null means the rule-based path handled this frame; it must not become 0,
    // which the UI would show as a genuinely terrible rep.
    expect(metricsFromWire(repFrame({ formConfidence: null })).formConfidence).toBeNull();
    expect(metricsFromWire(repFrame({ formConfidence: 0 })).formConfidence).toBe(0);
  });

  it('substitutes readable defaults for empty coaching fields', () => {
    const result = metricsFromWire(repFrame({ feedback: '', formName: '' }));
    expect(result.feedback).toBe('Keep going.');
    expect(result.formName).toBe('Correct');
  });

  it('defaults a missing angle to a straight limb rather than zero', () => {
    const result = metricsFromWire(repFrame({ angle: NaN }));
    expect(result.angle).toBe(180);
  });

  it('falls back to count when rawCount is absent', () => {
    expect(metricsFromWire(repFrame({ rawCount: null })).rawCount).toBe(7);
  });

  it('does not put the voice cue object into the rendered metrics', () => {
    // The live screen crashed once by rendering {label, text} directly. Voice
    // is spoken, never displayed, so it must not appear in this shape at all.
    const result = metricsFromWire(repFrame());
    expect(Object.values(result)).not.toContainEqual(
      expect.objectContaining({ label: expect.anything(), text: expect.anything() }),
    );
  });
});

describe('shouldRenderFrame', () => {
  const previous = { rawCount: 5, count: 5, state: 'UP' };

  it('renders immediately when a rep is counted', () => {
    const next = { rawCount: 6, count: 6, state: 'UP' };
    expect(shouldRenderFrame(next, previous, 0, 80)).toBe(true);
  });

  it('renders immediately when the movement phase changes', () => {
    const next = { rawCount: 5, count: 5, state: 'DOWN' };
    expect(shouldRenderFrame(next, previous, 0, 80)).toBe(true);
  });

  it('throttles frames that change nothing visible', () => {
    const next = { rawCount: 5, count: 5, state: 'UP' };
    expect(shouldRenderFrame(next, previous, 20, 80)).toBe(false);
  });

  it('still renders once the throttle window passes', () => {
    const next = { rawCount: 5, count: 5, state: 'UP' };
    expect(shouldRenderFrame(next, previous, 90, 80)).toBe(true);
  });

  it('renders the first frame of a session', () => {
    const next = { rawCount: 0, count: 0, state: 'idle' };
    expect(shouldRenderFrame(next, null, 0, 80)).toBe(true);
  });
});
