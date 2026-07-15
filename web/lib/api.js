const API_ORIGIN = process.env.NEXT_PUBLIC_API_ORIGIN || 'http://127.0.0.1:8001';
const API_PREFIX = '/v1';

export class ApiError extends Error {
  constructor(message, status, details = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

const TOKENS_KEY = 'coachvision.tokens';

export function loadTokens() {
  if (typeof window === 'undefined') return null;
  try {
    return JSON.parse(window.localStorage.getItem(TOKENS_KEY)) || null;
  } catch {
    return null;
  }
}

export function saveTokens(tokens) {
  if (typeof window === 'undefined') return;
  if (tokens) {
    window.localStorage.setItem(TOKENS_KEY, JSON.stringify(tokens));
  } else {
    window.localStorage.removeItem(TOKENS_KEY);
  }
}

async function rawRequest(path, { method = 'GET', body, accessToken } = {}) {
  const headers = { Accept: 'application/json' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;

  const response = await fetch(`${API_ORIGIN}${API_PREFIX}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  let payload = null;
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    payload = await response.json();
  }
  if (!response.ok) {
    const message =
      (payload && typeof payload.detail === 'string' && payload.detail) ||
      `Request failed (${response.status})`;
    throw new ApiError(message, response.status, payload);
  }
  return payload;
}

async function refreshTokens() {
  const tokens = loadTokens();
  if (!tokens?.refreshToken) return null;
  try {
    const pair = await rawRequest('/auth/refresh', {
      method: 'POST',
      body: { refresh_token: tokens.refreshToken },
    });
    const next = { accessToken: pair.access_token, refreshToken: pair.refresh_token };
    saveTokens(next);
    return next;
  } catch {
    saveTokens(null);
    return null;
  }
}

/** Authenticated request with automatic one-shot refresh on 401. */
export async function api(path, options = {}) {
  const tokens = loadTokens();
  try {
    return await rawRequest(path, { ...options, accessToken: tokens?.accessToken });
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) {
      const next = await refreshTokens();
      if (next) {
        return rawRequest(path, { ...options, accessToken: next.accessToken });
      }
    }
    throw err;
  }
}

export async function login(email, password) {
  const pair = await rawRequest('/auth/login', {
    method: 'POST',
    body: { email, password },
  });
  saveTokens({ accessToken: pair.access_token, refreshToken: pair.refresh_token });
  return api('/users/me');
}

export function logout() {
  saveTokens(null);
}
