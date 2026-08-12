import { apiRequest, type Schemas } from './client';

export type Invite = Schemas['InviteResponse'];
export type TrainerClient = Schemas['TrainerClientResponse'];
export type SessionResponse = Schemas['SessionResponse'];
export type ClientSessionDetail = Schemas['DailyReportSessionResponse'];
export type InviteAcceptResponse = Schemas['InviteAcceptResponse'];

// --- Trainer side ---

export type CreateInviteOptions = {
  email?: string | null;
  expiresInDays?: number;
};

export async function createInvite({
  email = null,
  expiresInDays = 7,
}: CreateInviteOptions = {}): Promise<Invite> {
  const body: Schemas['InviteCreateRequest'] = {
    email: email || null,
    expiresInDays,
  };
  return apiRequest<Invite>('/trainer/invites', {
    method: 'POST',
    auth: true,
    body,
  });
}

export async function listInvites(): Promise<Invite[]> {
  return apiRequest<Invite[]>('/trainer/invites', { auth: true });
}

export async function revokeInvite(inviteId: string): Promise<Invite> {
  return apiRequest<Invite>(`/trainer/invites/${inviteId}`, {
    method: 'DELETE',
    auth: true,
  });
}

export async function listClients(): Promise<TrainerClient[]> {
  return apiRequest<TrainerClient[]>('/trainer/clients', { auth: true });
}

export async function listClientSessions(clientId: string): Promise<SessionResponse[]> {
  return apiRequest<SessionResponse[]>(`/trainer/clients/${clientId}/sessions`, {
    auth: true,
  });
}

export async function getClientSessionDetail(
  clientId: string,
  sessionId: string,
): Promise<ClientSessionDetail> {
  return apiRequest<ClientSessionDetail>(
    `/trainer/clients/${clientId}/sessions/${sessionId}`,
    { auth: true },
  );
}

export async function endClientLink(clientId: string): Promise<void> {
  await apiRequest<void>(`/trainer/clients/${clientId}`, {
    method: 'DELETE',
    auth: true,
  });
}

// --- Client side ---

export async function acceptInvite(token: string): Promise<InviteAcceptResponse> {
  const body: Schemas['InviteAcceptRequest'] = { token };
  return apiRequest<InviteAcceptResponse>('/invites/accept', {
    method: 'POST',
    auth: true,
    body,
  });
}

export async function listMyTrainers(): Promise<TrainerClient[]> {
  return apiRequest<TrainerClient[]>('/me/trainers', { auth: true });
}
