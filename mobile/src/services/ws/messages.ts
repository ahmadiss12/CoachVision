/**
 * Wire types for the live workout WebSocket (`/v1/ws/live`).
 *
 * Mirrors `backend/coachvision/api/ws_live.py` and `backend/coachvision/
 * WS_CONTRACT.md`. REST payloads are generated from the OpenAPI spec, but
 * WebSocket frames are not covered by OpenAPI, so these are maintained by
 * hand and must be updated alongside the server.
 *
 * Two details are easy to get wrong, and both are load-bearing:
 *
 * 1. **Nullable, not optional.** The server builds every metrics key
 *    explicitly (`_metrics_message`), and Python `None` serialises to JSON
 *    `null`. So `"holdDurationSec" in payload` is always true and the value
 *    may be `null`. Declaring these `?:` would let `payload.holdDurationSec`
 *    read as `number` when it is really `null`.
 *
 * 2. **Reps and holds are different messages.** `measurementType`
 *    discriminates them: a squat carries `null` in every hold field, and a
 *    plank carries numbers. Modelling it as a union means reading
 *    `bestHoldSec` off a rep message is a compile error rather than a silent
 *    `null` in the UI.
 */

/** `SCHEMA_VERSION` in backend/coachvision/realtime/contract.py. */
export const SCHEMA_VERSION = 1;

/** Mirrors `WsErrorCode`. */
export type WsErrorCode =
  | 'INVALID_TOKEN'
  | 'BAD_MESSAGE'
  | 'NO_SESSION'
  | 'ALREADY_ACTIVE'
  | 'UNSUPPORTED_EXERCISE'
  | 'BAD_DIFFICULTY'
  | 'START_FAILED'
  | 'MODEL_MISSING'
  | 'BAD_FRAME';

export type Difficulty = 'beginner' | 'intermediate' | 'advanced';

/**
 * A spoken coaching cue.
 *
 * This is an object, not a string. Rendering it directly is what crashed the
 * live screen before this file existed (`Fix live view crash: render voice cue
 * text, not the {label, text} object`) -- read `.text`.
 */
export type VoiceCue = {
  /** De-duplication key. Not for display. */
  label: string;
  /** The phrase to speak or show. */
  text: string;
};

/** Normalised `[x, y, presence]` per landmark, MediaPipe ordering. */
export type PoseLandmark = readonly [x: number, y: number, presence: number];

/** Per-stage server timings in milliseconds. Keys vary by code path. */
export type ServerTimingMs = Record<string, number | string>;

/** Which model produced `formName`. */
export type FormSource = 'xgboost' | 'rule_based' | 'rule_based_fallback';

type MetricsBase = {
  schemaVersion: number;
  type: 'metrics';
  sessionId: string;
  /** What the UI shows: reps for dynamic exercises, seconds for holds. */
  count: number;
  /** The counter's own tally. For holds this is completed holds, not seconds. */
  rawCount: number | null;
  /** Exercise-specific FSM state, e.g. `UP`, `DOWN`, `HOLDING`. */
  state: string;
  /** Primary joint angle in degrees. */
  angle: number;
  /** Coaching text. Never null -- the server falls back to "Keep going.". */
  feedback: string;
  /** Progress through the current rep or hold, 0..1. */
  progress: number;
  formName: string;
  /** Only set when the XGBoost squat model ran; null on the rule-based path. */
  formConfidence: number | null;
  /** Per-class probabilities, same ordering as the model's class names. */
  formProbabilities: number[] | null;
  formSource: FormSource | null;
  /** Mean presence of the tracked joints, 0..1. */
  confidence: number;
  /** Null unless the client opted in with `includePose: true` on start. */
  pose: PoseLandmark[] | null;
  voice: VoiceCue | null;
  serverTimingMs: ServerTimingMs | null;
};

/** Metrics for a rep-counted exercise. Hold fields are always null. */
export type RepMetricsMessage = MetricsBase & {
  measurementType: 'reps';
  measurementLabel: 'REPS';
  holdDurationSec: null;
  totalHoldTimeSec: null;
  bestHoldSec: null;
  completedHolds: null;
};

/** Metrics for a hold exercise (plank, wall sit). `count` is seconds. */
export type HoldMetricsMessage = MetricsBase & {
  measurementType: 'hold';
  measurementLabel: 'SEC';
  /** Seconds held in the current hold. */
  holdDurationSec: number;
  /** Seconds held across the whole session, including the hold in progress. */
  totalHoldTimeSec: number;
  bestHoldSec: number;
  /** Holds that met the minimum duration. */
  completedHolds: number;
};

export type MetricsMessage = RepMetricsMessage | HoldMetricsMessage;

export type StartedMessage = {
  schemaVersion: number;
  type: 'started';
  sessionId: string;
};

export type NoPoseMessage = {
  schemaVersion: number;
  type: 'noPose';
  sessionId: string;
  serverTimingMs: ServerTimingMs | null;
};

export type ResetAckMessage = {
  schemaVersion: number;
  type: 'resetAck';
  sessionId: string;
};

/** Raw dispatcher export. Shape varies by exercise, so it stays unknown. */
export type SessionSummary = Record<string, unknown>;

export type EndedMessage = {
  schemaVersion: number;
  type: 'ended';
  sessionId: string;
  /** Null when the socket dropped mid-workout instead of ending cleanly. */
  summary: SessionSummary | null;
};

export type ErrorMessage = {
  schemaVersion: number;
  type: 'error';
  sessionId: string | null;
  message: string;
  code: WsErrorCode;
};

/** Every frame the server can send. Discriminated by `type`. */
export type ServerMessage =
  | StartedMessage
  | MetricsMessage
  | NoPoseMessage
  | ResetAckMessage
  | EndedMessage
  | ErrorMessage;

export type StartMessage = {
  type: 'start';
  exerciseName: string;
  difficulty: Difficulty;
  targetSets?: number;
  targetReps?: number;
  /** Links this socket to a session already created over REST. */
  sessionId?: string;
  /**
   * Echo landmarks back on every `metrics` frame. Defaults to false: the
   * client already has them and draws the skeleton locally, so echoing 33
   * landmarks per frame is wasted payload.
   */
  includePose?: boolean;
  externalLoadKg?: number;
  bodyWeightKg?: number;
  readinessContext?: Record<string, unknown>;
};

/** Landmarks from on-device MediaPipe. */
export type PoseMessage = {
  type: 'pose';
  sessionId: string;
  landmarks: PoseLandmark[] | { x: number; y: number; presence?: number }[];
  timestampMs?: number;
  clientInferenceMs?: number;
};

/** JPEG upload path, where the server runs pose detection itself. */
export type FrameMessage = {
  type: 'frame';
  sessionId: string;
  imageJpegBase64: string;
  timestampMs?: number;
};

export type ResetMessage = { type: 'reset'; sessionId: string };
export type EndMessage = { type: 'end'; sessionId: string };

export type ClientMessage =
  | StartMessage
  | PoseMessage
  | FrameMessage
  | ResetMessage
  | EndMessage;

/** True when the metrics frame is for a hold exercise (plank, wall sit). */
export function isHoldMetrics(
  metrics: MetricsMessage,
): metrics is HoldMetricsMessage {
  return metrics.measurementType === 'hold';
}
