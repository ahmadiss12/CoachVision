import { buildApiUrl } from './config';

let authTokens = null;

export class ApiError extends Error {
  constructor(message, status, details = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

export function setApiAuthTokens(tokens) {
  authTokens = tokens;
}

export function clearApiAuthTokens() {
  authTokens = null;
}

async function parseResponse(response) {
  // 204/205 carry no body, but FastAPI still labels them application/json.
  // Calling response.json() on an empty body throws, which would surface a
  // successful delete as a failure.
  if (response.status === 204 || response.status === 205) {
    return null;
  }
  const text = await response.text();
  if (!text) {
    return null;
  }
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }
  return text;
}

function getErrorMessage(payload, fallback) {
  if (!payload) {
    return fallback;
  }
  if (typeof payload === 'string') {
    return payload;
  }
  if (typeof payload.detail === 'string') {
    return payload.detail;
  }
  return fallback;
}

export async function apiRequest(path, options = {}) {
  const {
    method = 'GET',
    body,
    headers = {},
    auth = false,
    signal,
  } = options;
  const requestHeaders = {
    Accept: 'application/json',
    ...headers,
  };

  if (body !== undefined) {
    requestHeaders['Content-Type'] = 'application/json';
  }
  if (auth && authTokens?.accessToken) {
    requestHeaders.Authorization = `Bearer ${authTokens.accessToken}`;
  }

  const response = await fetch(buildApiUrl(path), {
    method,
    headers: requestHeaders,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal,
  });

  const payload = await parseResponse(response);
  if (!response.ok) {
    const fallback = `Request failed (${response.status})`;
    throw new ApiError(getErrorMessage(payload, fallback), response.status, payload);
  }
  return payload;
}
