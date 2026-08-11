import { apiRequest, type Schemas } from './client';

export type SessionResponse = Schemas['SessionResponse'];
export type SessionFeedbackResponse = Schemas['SessionFeedbackResponse'];

/**
 * The shape screens consume.
 *
 * The backend already returns camelCase for these endpoints (confirmed in the
 * generated schema), so this no longer normalises key casing -- it only fills
 * defaults so screens do not each repeat `?? 0`. The previous version also
 * accepted `exercise_id`, `total_reps` and friends; the generated types show
 * `SessionResponse` has no snake_case properties at all, so those branches
 * could never run.
 */
export type Session = {
  id: string;
  exerciseId: string;
  difficulty: string;
  status: string;
  targetSets: number;
  targetReps: number;
  externalLoadKg: number | null;
  bodyWeightKg: number | null;
  totalReps: number;
  avgFormScore: number | null;
  durationSeconds: number | null;
  createdAt: string;
  startedAt: string | null;
  endedAt: string | null;
};

export type SessionFeedback = {
  id: string;
  sessionId: string;
  exerciseId: string;
  overallRating: number;
  summaryText: string;
  errorsCount: number;
  topErrors: Schemas['FeedbackTopErrorItem'][];
  errorBreakdown: Record<string, unknown>;
  actionItems: Schemas['FeedbackActionItem'][];
  confidenceOverall: number | null;
  signalsUsed: Record<string, unknown>;
  version: string;
  generatedAt: string;
  updatedAt: string;
};

function toSessionShape(payload: SessionResponse): Session {
  return {
    id: payload.id,
    exerciseId: payload.exerciseId,
    difficulty: payload.difficulty,
    status: payload.status,
    targetSets: payload.targetSets ?? 0,
    targetReps: payload.targetReps ?? 0,
    externalLoadKg: payload.externalLoadKg ?? null,
    bodyWeightKg: payload.bodyWeightKg ?? null,
    totalReps: payload.totalReps ?? 0,
    avgFormScore: payload.avgFormScore ?? null,
    durationSeconds: payload.durationSeconds ?? null,
    createdAt: payload.createdAt,
    startedAt: payload.startedAt ?? null,
    endedAt: payload.endedAt ?? null,
  };
}

function toFeedbackShape(payload: SessionFeedbackResponse): SessionFeedback {
  return {
    id: payload.id,
    sessionId: payload.sessionId,
    exerciseId: payload.exerciseId,
    overallRating: payload.overallRating ?? 0,
    summaryText: payload.summaryText ?? '',
    errorsCount: payload.errorsCount ?? 0,
    topErrors: payload.topErrors ?? [],
    errorBreakdown: payload.errorBreakdown ?? {},
    actionItems: payload.actionItems ?? [],
    confidenceOverall: payload.confidenceOverall ?? null,
    signalsUsed: payload.signalsUsed ?? {},
    version: payload.version,
    generatedAt: payload.generatedAt,
    updatedAt: payload.updatedAt,
  };
}

export type CreateSessionOptions = {
  exerciseId: string;
  targetSets?: number;
  targetReps?: number;
  difficulty?: string;
  externalLoadKg?: number | null;
  bodyWeightKg?: number | null;
  assignmentId?: string | null;
};

export async function createSession({
  exerciseId,
  targetSets = 1,
  targetReps = 1,
  difficulty = 'beginner',
  externalLoadKg = null,
  bodyWeightKg = null,
  assignmentId = null,
}: CreateSessionOptions): Promise<Session> {
  const body: Schemas['CreateSessionRequest'] = {
    exerciseId,
    targetSets,
    targetReps,
    difficulty,
    externalLoadKg,
    bodyWeightKg,
    assignmentId,
  };
  const payload = await apiRequest<SessionResponse>('/sessions', {
    method: 'POST',
    auth: true,
    body,
  });
  return toSessionShape(payload);
}

export async function getSession(sessionId: string): Promise<Session> {
  const payload = await apiRequest<SessionResponse>(`/sessions/${sessionId}`, {
    auth: true,
  });
  return toSessionShape(payload);
}

export async function listSessions(): Promise<Session[]> {
  const payload = await apiRequest<SessionResponse[]>('/sessions', { auth: true });
  return Array.isArray(payload) ? payload.map(toSessionShape) : [];
}

export async function startSession(sessionId: string): Promise<Session> {
  const payload = await apiRequest<SessionResponse>(`/sessions/${sessionId}/start`, {
    method: 'POST',
    auth: true,
  });
  return toSessionShape(payload);
}

export type SessionSummaryInput = {
  totalReps?: number;
  avgFormScore?: number | null;
  externalLoadKg?: number | null;
  bodyWeightKg?: number | null;
};

export async function endSession(
  sessionId: string,
  summary: SessionSummaryInput = {},
): Promise<Session> {
  const body: Schemas['EndSessionRequest'] = {
    totalReps: summary.totalReps ?? 0,
    avgFormScore: summary.avgFormScore ?? null,
    externalLoadKg: summary.externalLoadKg ?? null,
    bodyWeightKg: summary.bodyWeightKg ?? null,
  };
  const payload = await apiRequest<SessionResponse>(`/sessions/${sessionId}/end`, {
    method: 'POST',
    auth: true,
    body,
  });
  return toSessionShape(payload);
}

export async function getSessionFeedback(sessionId: string): Promise<SessionFeedback> {
  const payload = await apiRequest<SessionFeedbackResponse>(
    `/sessions/${sessionId}/feedback`,
    { auth: true },
  );
  return toFeedbackShape(payload);
}
