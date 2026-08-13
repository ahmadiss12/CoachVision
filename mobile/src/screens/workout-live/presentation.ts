/**
 * Pure display logic for the live workout screen.
 *
 * Kept out of the component so it can be tested without rendering: these are
 * the rules that decide what colour the screen turns and what it says, and they
 * are the parts most likely to be wrong in a way a user notices.
 */

import type { LiveMetrics } from '../../state/live-metrics';

/** Connection state of the live socket. */
export type LiveStatus = 'idle' | 'connecting' | 'live' | 'paused' | 'demo' | 'error';

/** Loading state of the pose WebView. */
export type WebStatus = 'loading' | 'camera' | 'ready' | 'error';

export type FormTone = 'idle' | 'good' | 'warning' | 'danger';

export type ToneMeta = {
  label: string;
  /** Ionicons glyph name. */
  icon: string;
  color: string;
  soft: string;
  border: string;
};

export const FORM_TONE_META: Record<FormTone, ToneMeta> = {
  idle: {
    label: 'Ready',
    icon: 'scan-outline',
    color: '#38bdf8',
    soft: 'rgba(56, 189, 248, 0.16)',
    border: 'rgba(56, 189, 248, 0.36)',
  },
  good: {
    label: 'Good form',
    icon: 'checkmark-circle-outline',
    color: '#22c55e',
    soft: 'rgba(34, 197, 94, 0.18)',
    border: 'rgba(34, 197, 94, 0.42)',
  },
  warning: {
    label: 'Needs control',
    icon: 'alert-circle-outline',
    color: '#fbbf24',
    soft: 'rgba(251, 191, 36, 0.18)',
    border: 'rgba(251, 191, 36, 0.42)',
  },
  danger: {
    label: 'Fix form',
    icon: 'warning-outline',
    color: '#ef4444',
    soft: 'rgba(239, 68, 68, 0.18)',
    border: 'rgba(239, 68, 68, 0.48)',
  },
};

/** Phrases that mean the rep is unsafe or the athlete is out of frame. */
const DANGER_PHRASES = [
  'no person',
  'no pose',
  'not detected',
  'step back',
  'lost pose',
  'unsafe',
  'wrong',
  'broken',
  'collapse',
  'sag',
  "don't",
];

/** Phrases that mean the rep is happening but needs correcting. */
const WARNING_PHRASES = [
  'adjust',
  'shallow',
  'deeper',
  'lean',
  'chest',
  'knees',
  'heels',
  'control',
  'straighten',
  'align',
];

/** Below this the pose is too uncertain to coach from at all. */
const DANGER_CONFIDENCE = 0.35;
/** Below this the pose is usable but the athlete should be warned. */
const WARNING_CONFIDENCE = 0.6;

export function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(value)));
}

/**
 * Decide the screen's overall tone from the latest metrics.
 *
 * Connection problems outrank form: a camera error means the coaching text on
 * screen is stale, so it must not stay green.
 */
export function getFormTone(
  metrics: LiveMetrics | null | undefined,
  liveStatus: LiveStatus,
  webStatus: WebStatus,
): FormTone {
  if (webStatus === 'error' || liveStatus === 'error') {
    return 'danger';
  }
  if (!metrics || liveStatus === 'idle' || liveStatus === 'connecting') {
    return 'idle';
  }

  const feedback = String(metrics.feedback || '').toLowerCase();
  const formName = String(metrics.formName || '').toLowerCase();
  const combined = `${feedback} ${formName}`;
  const confidence = Number(metrics.confidence);
  // A confidence of exactly 0 means "not reported yet", not "no pose at all",
  // so it must not by itself turn the screen red.
  const hasConfidence = Number.isFinite(confidence) && confidence > 0;

  if (DANGER_PHRASES.some((phrase) => combined.includes(phrase))) {
    return 'danger';
  }
  if (hasConfidence && confidence < DANGER_CONFIDENCE) {
    return 'danger';
  }
  if (
    (hasConfidence && confidence < WARNING_CONFIDENCE) ||
    (formName !== '' && formName !== 'correct') ||
    WARNING_PHRASES.some((phrase) => combined.includes(phrase))
  ) {
    return 'warning';
  }
  return 'good';
}

/** Colour for a single line of coaching text. */
export function feedbackTone(message: string | null | undefined, fallback: string): string {
  if (!message) {
    return fallback;
  }
  const lower = message.toLowerCase();
  if (message.includes('!') || lower.includes("don't") || lower.includes('no pose')) {
    return '#f87171';
  }
  if (lower.includes('good') || lower.includes('great')) {
    return '#4ade80';
  }
  return '#fbbf24';
}

/** Short label for the status pill. Socket state wins over WebView state. */
export function getLiveStatusText(liveStatus: LiveStatus, webStatus: WebStatus): string {
  if (liveStatus === 'live') {
    return 'Live AI';
  }
  if (liveStatus === 'paused') {
    return 'Paused';
  }
  if (liveStatus === 'connecting') {
    return 'Connecting';
  }
  if (liveStatus === 'demo') {
    return 'Demo';
  }
  if (webStatus === 'ready') {
    return 'Ready';
  }
  if (webStatus === 'camera') {
    return 'Camera on';
  }
  if (webStatus === 'error') {
    return 'Camera issue';
  }
  return 'Loading AI';
}

export function getStatusColor(liveStatus: LiveStatus, webStatus: WebStatus): string {
  if (webStatus === 'error' || liveStatus === 'error') {
    return '#ef4444';
  }
  return liveStatus === 'live' ? '#22c55e' : '#fbbf24';
}
