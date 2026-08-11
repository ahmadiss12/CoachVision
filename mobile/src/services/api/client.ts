import { buildApiUrl } from './config';
import type { components } from './schema';

/**
 * Every response and request body the backend defines.
 *
 * `schema.d.ts` is generated from `backend/openapi.json`, which is exported
 * straight from the FastAPI app -- so these types are the server's own Pydantic
 * models, not a hand-kept copy that can drift. Regenerate with
 * `npm run generate:api`; CI fails if the committed file is stale.
 */
export type Schemas = components['schemas'];

export type AuthTokens = {
  accessToken: string;
  refreshToken?: string | null;
};

let authTokens: AuthTokens | null = null;

export class ApiError extends Error {
  readonly status: number;
  readonly details: unknown;

  constructor(message: string, status: number, details: unknown = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

export function setApiAuthTokens(tokens: AuthTokens | null): void {
  authTokens = tokens;
}

export function clearApiAuthTokens(): void {
  authTokens = null;
}

async function parseResponse(response: Response): Promise<unknown> {
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

function getErrorMessage(payload: unknown, fallback: string): string {
  if (!payload) {
    return fallback;
  }
  if (typeof payload === 'string') {
    return payload;
  }
  // FastAPI puts the message in `detail`, which is a string for HTTPException
  // and an array of validation errors for 422.
  if (typeof payload === 'object' && payload !== null) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === 'string') {
      return detail;
    }
  }
  return fallback;
}

export type ApiRequestOptions = {
  method?: 'GET' | 'POST' | 'PATCH' | 'PUT' | 'DELETE';
  body?: unknown;
  headers?: Record<string, string>;
  auth?: boolean;
  signal?: AbortSignal;
};

/**
 * Call the backend and return the parsed body.
 *
 * `Result` is the response type, taken from {@link Schemas} at the call site --
 * for example `apiRequest<Schemas['SessionResponse']>('/sessions/123')`. It is
 * an assertion, not a runtime check: it makes the *reading* of the payload
 * type-safe, so a renamed or removed server field fails to compile.
 */
export async function apiRequest<Result = unknown>(
  path: string,
  options: ApiRequestOptions = {},
): Promise<Result> {
  const { method = 'GET', body, headers = {}, auth = false, signal } = options;

  const requestHeaders: Record<string, string> = {
    Accept: 'application/json',
    ...headers,
  };

  if (body !== undefined) {
    requestHeaders['Content-Type'] = 'application/json';
  }
  if (auth && authTokens?.accessToken) {
    requestHeaders.Authorization = `Bearer ${authTokens.accessToken}`;
  }

  // Built conditionally rather than passing `body: undefined`: setting a key to
  // undefined is not the same as omitting it, and RequestInit.body does not
  // accept undefined as a value. A GET must carry no body key at all.
  const init: RequestInit = { method, headers: requestHeaders };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
  }
  if (signal) {
    init.signal = signal;
  }

  const response = await fetch(buildApiUrl(path), init);

  const payload = await parseResponse(response);
  if (!response.ok) {
    const fallback = `Request failed (${response.status})`;
    throw new ApiError(getErrorMessage(payload, fallback), response.status, payload);
  }
  return payload as Result;
}
