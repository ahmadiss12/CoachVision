/**
 * Messages exchanged with the pose WebView.
 *
 * The page in `pose-webview-html.ts` runs MediaPipe on-device and posts
 * landmark frames back to React Native. That is a second untyped boundary
 * alongside the WebSocket, and unlike the socket it is not documented anywhere
 * else -- these types are the contract.
 *
 * Both sides are ours, but the payload still arrives as a JSON string, so
 * {@link parseWebViewMessage} validates the discriminant rather than casting
 * blindly. A malformed frame from a page that failed to initialise should be
 * ignored, not crash the workout.
 */

import type { PoseLandmark } from '../../services/ws/messages';

/** The camera opened and is showing a preview. */
export type CameraReadyMessage = { type: 'cameraReady' };

/** MediaPipe finished loading and is producing landmarks. */
export type ModelReadyMessage = { type: 'modelReady' };

/** One detected pose. */
export type PoseFrameMessage = {
  type: 'pose';
  landmarks: PoseLandmark[] | { x: number; y: number; presence?: number }[];
  timestampMs?: number;
  /** How long on-device inference took, forwarded to the server for timing. */
  inferenceMs?: number;
};

/** A frame with no person in it. */
export type NoPoseMessage = { type: 'noPose' };

/** The page itself failed -- camera permission, model download, etc. */
export type WebErrorMessage = { type: 'webError'; message?: string };

export type WebViewMessage =
  | CameraReadyMessage
  | ModelReadyMessage
  | PoseFrameMessage
  | NoPoseMessage
  | WebErrorMessage;

/** Messages sent from React Native into the page. */
export type OverlayToneMessage = { type: 'overlayTone'; tone: string };
export type WebViewCommand = OverlayToneMessage;

const KNOWN_TYPES = new Set(['cameraReady', 'modelReady', 'pose', 'noPose', 'webError']);

/**
 * Parse a message posted by the WebView.
 *
 * Returns null for anything unusable -- invalid JSON, a missing or unknown
 * `type`, or a `pose` frame with no landmark array. Callers ignore null rather
 * than surfacing an error: a page still starting up can legitimately post
 * something we do not handle.
 */
export function parseWebViewMessage(raw: unknown): WebViewMessage | null {
  if (typeof raw !== 'string') {
    return null;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }

  if (typeof parsed !== 'object' || parsed === null) {
    return null;
  }

  const candidate = parsed as { type?: unknown; landmarks?: unknown };
  if (typeof candidate.type !== 'string' || !KNOWN_TYPES.has(candidate.type)) {
    return null;
  }

  // A pose frame without landmarks is the one malformed case that would
  // otherwise reach the socket and be sent to the server.
  if (candidate.type === 'pose' && !Array.isArray(candidate.landmarks)) {
    return null;
  }

  return parsed as WebViewMessage;
}
