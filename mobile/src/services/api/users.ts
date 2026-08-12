import { apiRequest, type Schemas } from './client';

/**
 * Like auth, the user profile endpoints are snake_case (`display_name`,
 * `height_cm`, `date_of_birth`) while most of the API is camelCase. These types
 * make that explicit rather than leaving callers to guess.
 */
export type UserMe = Schemas['UserMeResponse'];
export type UpdateUserMeRequest = Schemas['UpdateUserMeRequest'];
export type BodyMetric = Schemas['BodyMetricResponse'];
export type BodyMetricCreateRequest = Schemas['BodyMetricCreateRequest'];

export async function getCurrentUser(): Promise<UserMe> {
  return apiRequest<UserMe>('/users/me', { auth: true });
}

export async function updateCurrentUser(payload: UpdateUserMeRequest): Promise<UserMe> {
  return apiRequest<UserMe>('/users/me', {
    method: 'PATCH',
    auth: true,
    body: payload,
  });
}

/**
 * Permanently delete the signed-in account. The password is re-checked
 * server-side, so a stolen phone or leaked token cannot destroy an account.
 */
export async function deleteCurrentUser(password: string): Promise<void> {
  const body: Schemas['DeleteUserMeRequest'] = { password };
  await apiRequest<void>('/users/me', {
    method: 'DELETE',
    auth: true,
    body,
  });
}

export async function listBodyMetricEntries(): Promise<BodyMetric[]> {
  return apiRequest<BodyMetric[]>('/users/me/body-metrics', { auth: true });
}

export async function createBodyMetricEntry(
  payload: BodyMetricCreateRequest,
): Promise<BodyMetric> {
  return apiRequest<BodyMetric>('/users/me/body-metrics', {
    method: 'POST',
    auth: true,
    body: payload,
  });
}
