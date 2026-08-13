import { parseWebViewMessage } from '../webview-bridge';

describe('parseWebViewMessage', () => {
  it('parses the lifecycle messages', () => {
    expect(parseWebViewMessage('{"type":"cameraReady"}')).toEqual({ type: 'cameraReady' });
    expect(parseWebViewMessage('{"type":"modelReady"}')).toEqual({ type: 'modelReady' });
    expect(parseWebViewMessage('{"type":"noPose"}')).toEqual({ type: 'noPose' });
  });

  it('parses a pose frame with its landmarks', () => {
    const raw = JSON.stringify({
      type: 'pose',
      landmarks: [[0.5, 0.5, 0.9]],
      timestampMs: 1700000000000,
      inferenceMs: 12.5,
    });
    const parsed = parseWebViewMessage(raw);
    expect(parsed).toMatchObject({ type: 'pose', inferenceMs: 12.5 });
  });

  it('parses a web error with its message', () => {
    const parsed = parseWebViewMessage('{"type":"webError","message":"camera denied"}');
    expect(parsed).toEqual({ type: 'webError', message: 'camera denied' });
  });

  it('rejects a pose frame with no landmark array', () => {
    // This is the one malformed case that would otherwise be forwarded to the
    // server as an empty pose.
    expect(parseWebViewMessage('{"type":"pose"}')).toBeNull();
    expect(parseWebViewMessage('{"type":"pose","landmarks":null}')).toBeNull();
    expect(parseWebViewMessage('{"type":"pose","landmarks":"nope"}')).toBeNull();
  });

  it('rejects malformed input rather than throwing', () => {
    expect(parseWebViewMessage('not json')).toBeNull();
    expect(parseWebViewMessage('')).toBeNull();
    expect(parseWebViewMessage('null')).toBeNull();
    expect(parseWebViewMessage('[]')).toBeNull();
    expect(parseWebViewMessage('123')).toBeNull();
  });

  it('rejects non-string input', () => {
    expect(parseWebViewMessage(undefined)).toBeNull();
    expect(parseWebViewMessage({ type: 'pose' })).toBeNull();
  });

  it('ignores message types it does not handle', () => {
    // A page still starting up may post things we do not know about; that is
    // not an error worth surfacing to the athlete.
    expect(parseWebViewMessage('{"type":"somethingNew"}')).toBeNull();
    expect(parseWebViewMessage('{"nope":1}')).toBeNull();
  });
});
