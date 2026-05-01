import { apiRequest } from './client';

function normalizeTokenPair(payload) {
  return {
    accessToken: payload.access_token ?? payload.accessToken,
    refreshToken: payload.refresh_token ?? payload.refreshToken,
    tokenType: payload.token_type ?? payload.tokenType ?? 'bearer',
  };
}

export async function registerUser({ email, password, displayName }) {
  const payload = await apiRequest('/auth/register', {
    method: 'POST',
    body: {
      email,
      password,
      display_name: displayName,
    },
  });
  return normalizeTokenPair(payload);
}

export async function loginUser({ email, password }) {
  const payload = await apiRequest('/auth/login', {
    method: 'POST',
    body: { email, password },
  });
  return normalizeTokenPair(payload);
}

export async function refreshAuthToken(refreshToken) {
  const payload = await apiRequest('/auth/refresh', {
    method: 'POST',
    body: { refresh_token: refreshToken },
  });
  return normalizeTokenPair(payload);
}

export async function logoutUser() {
  await apiRequest('/auth/logout', {
    method: 'POST',
    auth: true,
  });
}
