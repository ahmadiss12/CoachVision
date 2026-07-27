import { apiRequest } from './client';

export async function getCurrentUser() {
  return apiRequest('/users/me', { auth: true });
}

export async function updateCurrentUser(payload) {
  return apiRequest('/users/me', {
    method: 'PATCH',
    auth: true,
    body: payload,
  });
}

/**
 * Permanently delete the signed-in account. The password is re-checked
 * server-side, so a stolen phone or leaked token cannot destroy an account.
 */
export async function deleteCurrentUser(password) {
  return apiRequest('/users/me', {
    method: 'DELETE',
    auth: true,
    body: { password },
  });
}

export async function listBodyMetricEntries() {
  return apiRequest('/users/me/body-metrics', { auth: true });
}

export async function createBodyMetricEntry(payload) {
  return apiRequest('/users/me/body-metrics', {
    method: 'POST',
    auth: true,
    body: payload,
  });
}
