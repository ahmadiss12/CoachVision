import { apiRequest } from './client';

export async function listAllUsers() {
  return apiRequest('/admin/users', { auth: true });
}

export async function changeUserRole(userId, role) {
  return apiRequest(`/admin/users/${userId}/role`, {
    method: 'PATCH',
    auth: true,
    body: { role },
  });
}

export async function deleteUser(userId) {
  return apiRequest(`/admin/users/${userId}`, {
    method: 'DELETE',
    auth: true,
  });
}

export async function getPlatformStats() {
  return apiRequest('/admin/stats', { auth: true });
}
