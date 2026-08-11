/**
 * Compile-time tests for the live WebSocket contract.
 *
 * `@ts-expect-error` fails the build when the line below it *stops* being an
 * error, so each one pins a mistake the types must keep rejecting. Loosen a
 * type in messages.ts and `tsc` reports an unused directive here.
 *
 * Nothing runs; `npm run typecheck` is the test.
 */

import {
  isHoldMetrics,
  type HoldMetricsMessage,
  type MetricsMessage,
  type RepMetricsMessage,
  type ServerMessage,
  type StartMessage,
} from './messages';

declare const metrics: MetricsMessage;
declare const repMetrics: RepMetricsMessage;
declare const holdMetrics: HoldMetricsMessage;
declare const message: ServerMessage;

// --- The crash this migration exists to prevent -----------------------------
// "Fix live view crash: render voice cue text, not the {label, text} object"

// @ts-expect-error voice is a VoiceCue object, never a string.
export const voiceAsString: string = metrics.voice;

// @ts-expect-error voice is null whenever there is nothing to say.
export const voiceAlwaysPresent: { label: string; text: string } = metrics.voice;

export const voiceText: string | undefined = metrics.voice?.text;

// --- Reps and holds are different messages ---------------------------------

// @ts-expect-error hold fields are null on a rep-counted exercise.
export const bestHoldOnReps: number = repMetrics.bestHoldSec;

export const bestHoldOnHold: number = holdMetrics.bestHoldSec;

// @ts-expect-error a rep message cannot claim to be a hold.
export const wrongLabel: RepMetricsMessage['measurementLabel'] = 'SEC';

// The type guard is what makes hold fields readable.
export function holdSeconds(m: MetricsMessage): number {
  if (isHoldMetrics(m)) {
    return m.totalHoldTimeSec;
  }
  return 0;
}

// Narrowing on the discriminant directly works too.
export function holdSecondsByDiscriminant(m: MetricsMessage): number {
  return m.measurementType === 'hold' ? m.bestHoldSec : 0;
}

// --- Fields that are null on some code paths -------------------------------

// @ts-expect-error formConfidence is null unless the XGBoost model ran.
export const confidencePercent: number = metrics.formConfidence * 100;

export const safeConfidence: number = (metrics.formConfidence ?? 0) * 100;

// @ts-expect-error pose is null unless the client asked for it on start.
export const firstLandmark = metrics.pose[0];

// --- Optional vs nullable ---------------------------------------------------
// The server writes every metrics key explicitly, so these arrive as null
// rather than being absent. Declaring them optional would be wrong.

export const holdIsNullNotMissing: null = repMetrics.holdDurationSec;

// @ts-expect-error `undefined` never appears on the wire; absent means null.
export const holdIsUndefined: undefined = repMetrics.holdDurationSec;

// --- Message discrimination -------------------------------------------------

export function sessionIdOf(m: ServerMessage): string | null {
  switch (m.type) {
    case 'started':
    case 'metrics':
    case 'noPose':
    case 'resetAck':
    case 'ended':
      return m.sessionId;
    case 'error':
      return m.sessionId;
  }
}

// @ts-expect-error 'summary' only exists on the 'ended' message.
export const summaryAnywhere = message.summary;

// --- Client messages --------------------------------------------------------

export const validStart: StartMessage = {
  type: 'start',
  exerciseName: 'squat',
  difficulty: 'intermediate',
  includePose: false,
};

// Kept on one line each: `@ts-expect-error` only covers the line immediately
// after it, and object-literal errors are reported on the offending property.

// @ts-expect-error difficulty is a fixed set, not any string.
export const badDifficulty: StartMessage = { type: 'start', exerciseName: 'squat', difficulty: 'expert' };

// @ts-expect-error exerciseName is required.
export const missingExercise: StartMessage = { type: 'start', difficulty: 'intermediate' };
