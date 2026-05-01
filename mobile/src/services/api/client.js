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
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }
  const text = await response.text();
  return text || null;
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
