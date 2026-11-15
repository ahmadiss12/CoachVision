import { apiRequest } from './client';

export async function getEntitlements() {
  return apiRequest('/me/entitlements', { auth: true });
}
