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
