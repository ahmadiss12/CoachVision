import { buildWsUrl } from '../api/config';
import type {
  ClientMessage,
  EndedMessage,
  ErrorMessage,
  MetricsMessage,
  NoPoseMessage,
  ServerMessage,
  StartedMessage,
} from './messages';

export type LiveSessionHandlers = {
  onStarted?: (payload: StartedMessage) => void;
  onMetrics?: (payload: MetricsMessage) => void;
  onNoPose?: (payload: NoPoseMessage) => void;
  onEnded?: (payload: EndedMessage) => void;
  onError?: (message: string, code?: ErrorMessage['code']) => void;
};

export type LiveSessionOptions = LiveSessionHandlers & {
  accessToken: string;
};

export type LiveSessionSocket = {
  connect: () => Promise<void>;
  /** Returns false when the socket is closed or the send buffer is backed up. */
  send: (payload: ClientMessage) => boolean;
  close: () => void;
};

/** Above this, the network is not keeping up and new frames are dropped. */
const MAX_BUFFERED_BYTES = 1_000_000;

export function createLiveSessionSocket({
  accessToken,
  onMetrics,
  onStarted,
  onEnded,
  onNoPose,
  onError,
}: LiveSessionOptions): LiveSessionSocket {
  let socket: WebSocket | null = null;
  let opened = false;

  const connect = () =>
    new Promise<void>((resolve, reject) => {
      try {
        const url = buildWsUrl('/ws/live', { token: accessToken });
        socket = new WebSocket(url);
      } catch (error) {
        reject(error);
        return;
      }

      socket.onopen = () => {
        opened = true;
        resolve();
      };

      socket.onerror = () => {
        if (!opened) {
          reject(new Error('Unable to connect to live workout websocket.'));
          return;
        }
        onError?.('Live workout websocket error.');
      };

      socket.onmessage = (event: MessageEvent) => {
        try {
          // The server is the only writer on this socket, but it is still
          // untrusted input as far as the type system is concerned: the cast
          // asserts the contract in messages.ts, it does not verify it.
          const payload = JSON.parse(String(event.data)) as ServerMessage;
          switch (payload.type) {
            case 'started':
              onStarted?.(payload);
              break;
            case 'metrics':
              onMetrics?.(payload);
              break;
            case 'noPose':
              onNoPose?.(payload);
              break;
            case 'ended':
              onEnded?.(payload);
              break;
            case 'error':
              onError?.(payload.message || 'Live session error.', payload.code);
              break;
            default:
              break;
          }
        } catch {
          onError?.('Received invalid realtime payload.');
        }
      };

      socket.onclose = () => {
        if (!opened) {
          reject(new Error('Live websocket closed during setup.'));
        }
      };
    });

  const send = (payload: ClientMessage): boolean => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return false;
    }
    if (socket.bufferedAmount > MAX_BUFFERED_BYTES) {
      return false;
    }
    socket.send(JSON.stringify(payload));
    return true;
  };

  const close = () => {
    if (socket && socket.readyState <= WebSocket.OPEN) {
      socket.close();
    }
  };

  return {
    connect,
    send,
    close,
  };
}
