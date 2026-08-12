import { apiRequest, type Schemas } from './client';

/**
 * The auth endpoints are the snake_case corner of the API: `TokenPair` sends
 * `access_token`, while most of the rest of the API is camelCase. This module
 * is where that boundary is crossed -- everything downstream sees camelCase.
 *
 * The previous version also accepted `payload.accessToken`; the generated
 * schema shows `TokenPair` has no camelCase properties, so that branch could
 * never run.
 */
export type TokenPairResponse = Schemas['TokenPair'];

export type AuthTokenPair = {
  accessToken: string;
  refreshToken: string;
  tokenType: string;
};

function normalizeTokenPair(payload: TokenPairResponse): AuthTokenPair {
  return {
    accessToken: payload.access_token,
    refreshToken: payload.refresh_token,
    tokenType: payload.token_type ?? 'bearer',
  };
}

export type RegisterOptions = {
  email: string;
  password: string;
  displayName: string;
  role?: string;
};

export async function registerUser({
  email,
  password,
  displayName,
  role = 'client',
}: RegisterOptions): Promise<AuthTokenPair> {
  const body: Schemas['RegisterRequest'] = {
    email,
    password,
    display_name: displayName,
    role,
  };
  const payload = await apiRequest<TokenPairResponse>('/auth/register', {
    method: 'POST',
    body,
  });
  return normalizeTokenPair(payload);
}

export type LoginOptions = {
  email: string;
  password: string;
};

export async function loginUser({ email, password }: LoginOptions): Promise<AuthTokenPair> {
  const payload = await apiRequest<TokenPairResponse>('/auth/login', {
    method: 'POST',
    body: { email, password },
  });
  return normalizeTokenPair(payload);
}

export async function refreshAuthToken(refreshToken: string): Promise<AuthTokenPair> {
  const body: Schemas['RefreshRequest'] = { refresh_token: refreshToken };
  const payload = await apiRequest<TokenPairResponse>('/auth/refresh', {
    method: 'POST',
    body,
  });
  return normalizeTokenPair(payload);
}

export async function logoutUser(): Promise<void> {
  await apiRequest('/auth/logout', {
    method: 'POST',
    auth: true,
  });
}
