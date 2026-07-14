import { apiRequest } from './client';

// --- Trainer side ---

export async function createInvite({ email = null, expiresInDays = 7 } = {}) {
  return apiRequest('/trainer/invites', {
    method: 'POST',
    auth: true,
    body: { email: email || null, expiresInDays },
  });
}

export async function listInvites() {
  return apiRequest('/trainer/invites', { auth: true });
}

export async function revokeInvite(inviteId) {
  return apiRequest(`/trainer/invites/${inviteId}`, {
    method: 'DELETE',
    auth: true,
  });
}

export async function listClients() {
  return apiRequest('/trainer/clients', { auth: true });
}

export async function listClientSessions(clientId) {
  return apiRequest(`/trainer/clients/${clientId}/sessions`, { auth: true });
}

export async function getClientSessionDetail(clientId, sessionId) {
  return apiRequest(`/trainer/clients/${clientId}/sessions/${sessionId}`, { auth: true });
}

export async function endClientLink(clientId) {
  return apiRequest(`/trainer/clients/${clientId}`, {
    method: 'DELETE',
    auth: true,
  });
}

// --- Client side ---

export async function acceptInvite(token) {
  return apiRequest('/invites/accept', {
    method: 'POST',
    auth: true,
    body: { token },
  });
}

export async function listMyTrainers() {
  return apiRequest('/me/trainers', { auth: true });
}
