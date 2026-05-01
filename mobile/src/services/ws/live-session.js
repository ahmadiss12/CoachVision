import { buildWsUrl } from '../api/config';

export function createLiveSessionSocket({
  accessToken,
  onMetrics,
  onStarted,
  onEnded,
  onNoPose,
  onError,
}) {
  let socket = null;
  let opened = false;

  const connect = () =>
    new Promise((resolve, reject) => {
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

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
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
              onError?.(payload.message || 'Live session error.');
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

  const send = (payload) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
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
