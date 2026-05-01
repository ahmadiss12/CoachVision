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
