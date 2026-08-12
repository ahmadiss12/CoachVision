import { apiRequest, type Schemas } from './client';

export type AdminUser = Schemas['AdminUserResponse'];
export type AdminStats = Schemas['AdminStatsResponse'];

export async function listAllUsers(): Promise<AdminUser[]> {
  return apiRequest<AdminUser[]>('/admin/users', { auth: true });
}

export async function changeUserRole(userId: string, role: string): Promise<AdminUser> {
  const body: Schemas['UpdateRoleRequest'] = { role };
  return apiRequest<AdminUser>(`/admin/users/${userId}/role`, {
    method: 'PATCH',
    auth: true,
    body,
  });
}

export async function deleteUser(userId: string): Promise<void> {
  await apiRequest<void>(`/admin/users/${userId}`, {
    method: 'DELETE',
    auth: true,
  });
}

export async function getPlatformStats(): Promise<AdminStats> {
  return apiRequest<AdminStats>('/admin/stats', { auth: true });
}
