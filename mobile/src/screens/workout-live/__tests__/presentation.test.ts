import type { LiveMetrics } from '../../../state/live-metrics';
import {
  clampPercent,
  feedbackTone,
  getFormTone,
  getLiveStatusText,
  getStatusColor,
} from '../presentation';

const FALLBACK = '#888888';

function metrics(overrides: Partial<LiveMetrics> = {}): LiveMetrics {
  return {
    count: 5,
    rawCount: 5,
    measurementType: 'reps',
    measurementLabel: 'REPS',
    state: 'UP',
    angle: 170,
    feedback: 'Keep going.',
    progress: 0.5,
    formName: 'Correct',
    formConfidence: 0.9,
    formSource: 'xgboost',
    confidence: 0.9,
    ...overrides,
  };
}

describe('clampPercent', () => {
  it('rounds into the 0-100 range', () => {
    expect(clampPercent(42.4)).toBe(42);
    expect(clampPercent(42.6)).toBe(43);
  });

  it('clamps out-of-range values', () => {
    expect(clampPercent(-30)).toBe(0);
    expect(clampPercent(180)).toBe(100);
  });

  it('treats non-finite input as zero rather than full', () => {
    // A nonsense progress value should read as "no progress", not as a
    // completed rep -- so Infinity is rejected before the clamp, not clamped.
    expect(clampPercent(NaN)).toBe(0);
    expect(clampPercent(Infinity)).toBe(0);
    expect(clampPercent(-Infinity)).toBe(0);
  });
});

describe('getFormTone', () => {
  it('is idle before a session is producing metrics', () => {
    expect(getFormTone(null, 'idle', 'ready')).toBe('idle');
    expect(getFormTone(metrics(), 'connecting', 'ready')).toBe('idle');
  });

  it('is good for a clean rep', () => {
    expect(getFormTone(metrics(), 'live', 'ready')).toBe('good');
  });

  it('warns when the classifier reports a form other than Correct', () => {
    expect(getFormTone(metrics({ formName: 'Shallow' }), 'live', 'ready')).toBe('warning');
  });

  it('warns on coaching phrases that mean "fix this"', () => {
    for (const feedback of ['Go deeper', 'Straighten your back', 'Knees caving in']) {
      expect(getFormTone(metrics({ feedback }), 'live', 'ready')).toBe('warning');
    }
  });

  it('escalates to danger when the athlete is out of frame', () => {
    const out = metrics({ feedback: 'No person detected - step back' });
    expect(getFormTone(out, 'live', 'ready')).toBe('danger');
  });

  it('escalates to danger on very low pose confidence', () => {
    expect(getFormTone(metrics({ confidence: 0.2 }), 'live', 'ready')).toBe('danger');
  });

  it('warns on middling pose confidence', () => {
    expect(getFormTone(metrics({ confidence: 0.5 }), 'live', 'ready')).toBe('warning');
  });

  it('does not treat an unreported confidence as no pose', () => {
    // 0 means "not measured yet", which must not turn the screen red on the
    // first frame of a session.
    expect(getFormTone(metrics({ confidence: 0 }), 'live', 'ready')).toBe('good');
  });

  it('lets a connection failure override good-looking metrics', () => {
    expect(getFormTone(metrics(), 'live', 'error')).toBe('danger');
    expect(getFormTone(metrics(), 'error', 'ready')).toBe('danger');
  });
});

describe('feedbackTone', () => {
  it('falls back when there is nothing to say', () => {
    expect(feedbackTone('', FALLBACK)).toBe(FALLBACK);
    expect(feedbackTone(null, FALLBACK)).toBe(FALLBACK);
  });

  it('is red for urgent messages', () => {
    expect(feedbackTone('Stop!', FALLBACK)).toBe('#f87171');
    expect(feedbackTone("Don't lock your knees", FALLBACK)).toBe('#f87171');
    expect(feedbackTone('No pose detected', FALLBACK)).toBe('#f87171');
  });

  it('is green for praise', () => {
    expect(feedbackTone('Good rep', FALLBACK)).toBe('#4ade80');
    expect(feedbackTone('Great depth', FALLBACK)).toBe('#4ade80');
  });

  it('is amber for ordinary coaching', () => {
    expect(feedbackTone('Go a little deeper', FALLBACK)).toBe('#fbbf24');
  });
});

describe('getLiveStatusText', () => {
  it('reports socket state ahead of webview state', () => {
    expect(getLiveStatusText('live', 'loading')).toBe('Live AI');
    expect(getLiveStatusText('paused', 'ready')).toBe('Paused');
    expect(getLiveStatusText('demo', 'ready')).toBe('Demo');
  });

  it('falls back to webview state when the socket is idle', () => {
    expect(getLiveStatusText('idle', 'ready')).toBe('Ready');
    expect(getLiveStatusText('idle', 'camera')).toBe('Camera on');
    expect(getLiveStatusText('idle', 'error')).toBe('Camera issue');
    expect(getLiveStatusText('idle', 'loading')).toBe('Loading AI');
  });
});

describe('getStatusColor', () => {
  it('is green only while genuinely live', () => {
    expect(getStatusColor('live', 'ready')).toBe('#22c55e');
    expect(getStatusColor('paused', 'ready')).toBe('#fbbf24');
  });

  it('is red on either kind of failure', () => {
    expect(getStatusColor('error', 'ready')).toBe('#ef4444');
    expect(getStatusColor('live', 'error')).toBe('#ef4444');
  });
});
