import { apiRequest } from './client';

// --- Trainer side ---

export async function createProgram(payload) {
  return apiRequest('/trainer/programs', {
    method: 'POST',
    auth: true,
    body: payload,
  });
}

export async function listPrograms() {
  return apiRequest('/trainer/programs', { auth: true });
}

export async function deleteProgram(programId) {
  return apiRequest(`/trainer/programs/${programId}`, {
    method: 'DELETE',
    auth: true,
  });
}

export async function assignProgram(programId, clientId, startDate = null) {
  return apiRequest(`/trainer/programs/${programId}/assign`, {
    method: 'POST',
    auth: true,
    body: { clientId, startDate },
  });
}

export async function listClientAssignments(clientId) {
  return apiRequest(`/trainer/clients/${clientId}/assignments`, { auth: true });
}

export async function cancelAssignment(assignmentId) {
  return apiRequest(`/trainer/assignments/${assignmentId}`, {
    method: 'DELETE',
    auth: true,
  });
}

// --- Client side ---

export async function getTodayPlan() {
  return apiRequest('/me/plan/today', { auth: true });
}
