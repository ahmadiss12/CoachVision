import { apiRequest, type Schemas } from './client';

export type Program = Schemas['ProgramResponse'];
export type ProgramCreateRequest = Schemas['ProgramCreateRequest'];
export type Assignment = Schemas['AssignmentResponse'];
export type TodayPlan = Schemas['TodayPlanResponse'];

// --- Trainer side ---

export async function createProgram(payload: ProgramCreateRequest): Promise<Program> {
  return apiRequest<Program>('/trainer/programs', {
    method: 'POST',
    auth: true,
    body: payload,
  });
}

export async function listPrograms(): Promise<Program[]> {
  return apiRequest<Program[]>('/trainer/programs', { auth: true });
}

export async function deleteProgram(programId: string): Promise<void> {
  await apiRequest<void>(`/trainer/programs/${programId}`, {
    method: 'DELETE',
    auth: true,
  });
}

export async function assignProgram(
  programId: string,
  clientId: string,
  startDate: string | null = null,
): Promise<Assignment> {
  const body: Schemas['AssignProgramRequest'] = { clientId, startDate };
  return apiRequest<Assignment>(`/trainer/programs/${programId}/assign`, {
    method: 'POST',
    auth: true,
    body,
  });
}

export async function listClientAssignments(clientId: string): Promise<Assignment[]> {
  return apiRequest<Assignment[]>(`/trainer/clients/${clientId}/assignments`, {
    auth: true,
  });
}

export async function cancelAssignment(assignmentId: string): Promise<void> {
  await apiRequest<void>(`/trainer/assignments/${assignmentId}`, {
    method: 'DELETE',
    auth: true,
  });
}

// --- Client side ---

export async function getTodayPlan(): Promise<TodayPlan> {
  return apiRequest<TodayPlan>('/me/plan/today', { auth: true });
}
