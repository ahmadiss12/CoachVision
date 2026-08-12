import { apiRequest, type Schemas } from './client';
import type { SessionFeedback } from './sessions';

export type DailyReportResponse = Schemas['DailyReportResponse'];
export type DailyReportSessionResponse = Schemas['DailyReportSessionResponse'];
export type SessionFeedbackResponse = Schemas['SessionFeedbackResponse'];
export type ReportRepStatsResponse = Schemas['ReportRepStatsResponse'];

/**
 * The daily report schemas are entirely camelCase, so the snake_case fallbacks
 * this module used to carry on roughly forty fields could never run. What
 * remains is default-filling, so screens do not each repeat `?? 0`.
 */

export type RepStats = {
  repCount: number;
  avgRangeOfMotion: number | null;
  avgRepDurationMs: number | null;
  avgFormScore: number | null;
  depthQualityCounts: Record<string, number>;
};

export type ReportSession = {
  sessionId: string;
  exerciseId: string;
  exerciseName: string;
  difficulty: string;
  targetSets: number;
  targetReps: number;
  totalReps: number;
  avgFormScore: number | null;
  durationSeconds: number | null;
  startedAt: string | null;
  endedAt: string | null;
  externalLoadKg: number | null;
  bodyWeightKg: number | null;
  usesExternalLoad: boolean;
  estimatedVolumeLoadKg: number | null;
  feedback: SessionFeedback | null;
  repStats: RepStats;
};

export type DailyReport = {
  generatedAt: string;
  user: {
    id: string;
    email: string;
    displayName: string;
  };
  summary: {
    date: string;
    timezone: string;
    totalSessions: number;
    totalReps: number;
    totalDurationSeconds: number;
    avgFormScore: number | null;
    totalExternalLoadVolumeKg: number | null;
  };
  sessions: ReportSession[];
};

function toFeedbackShape(
  payload: SessionFeedbackResponse | null | undefined,
): SessionFeedback | null {
  if (!payload) {
    return null;
  }
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

function toRepStatsShape(payload: ReportRepStatsResponse | undefined): RepStats {
  return {
    repCount: payload?.repCount ?? 0,
    avgRangeOfMotion: payload?.avgRangeOfMotion ?? null,
    avgRepDurationMs: payload?.avgRepDurationMs ?? null,
    avgFormScore: payload?.avgFormScore ?? null,
    depthQualityCounts: payload?.depthQualityCounts ?? {},
  };
}

function toReportSessionShape(payload: DailyReportSessionResponse): ReportSession {
  return {
    sessionId: payload.sessionId,
    exerciseId: payload.exerciseId,
    exerciseName: payload.exerciseName ?? 'Workout',
    difficulty: payload.difficulty ?? '',
    targetSets: payload.targetSets ?? 1,
    targetReps: payload.targetReps ?? 1,
    totalReps: payload.totalReps ?? 0,
    avgFormScore: payload.avgFormScore ?? null,
    durationSeconds: payload.durationSeconds ?? null,
    startedAt: payload.startedAt ?? null,
    endedAt: payload.endedAt ?? null,
    externalLoadKg: payload.externalLoadKg ?? null,
    bodyWeightKg: payload.bodyWeightKg ?? null,
    usesExternalLoad: payload.usesExternalLoad ?? false,
    estimatedVolumeLoadKg: payload.estimatedVolumeLoadKg ?? null,
    feedback: toFeedbackShape(payload.feedback),
    repStats: toRepStatsShape(payload.repStats),
  };
}

function toDailyReportShape(payload: DailyReportResponse): DailyReport {
  const summary = payload.summary;
  const user = payload.user;
  return {
    generatedAt: payload.generatedAt,
    user: {
      id: user.id,
      email: user.email ?? '',
      displayName: user.displayName ?? 'Athlete',
    },
    summary: {
      date: summary.date,
      timezone: summary.timezone ?? 'UTC',
      totalSessions: summary.totalSessions ?? 0,
      totalReps: summary.totalReps ?? 0,
      totalDurationSeconds: summary.totalDurationSeconds ?? 0,
      avgFormScore: summary.avgFormScore ?? null,
      totalExternalLoadVolumeKg: summary.totalExternalLoadVolumeKg ?? null,
    },
    sessions: Array.isArray(payload.sessions)
      ? payload.sessions.map(toReportSessionShape)
      : [],
  };
}

export type GetDailyReportOptions = {
  date?: string;
};

export async function getDailyReport({
  date,
}: GetDailyReportOptions = {}): Promise<DailyReport> {
  const query = date ? `?date=${encodeURIComponent(date)}` : '';
  const payload = await apiRequest<DailyReportResponse>(`/reports/daily${query}`, {
    auth: true,
  });
  return toDailyReportShape(payload);
}
