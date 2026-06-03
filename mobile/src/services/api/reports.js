import { apiRequest } from './client';

function toFeedbackShape(payload) {
  if (!payload) {
    return null;
  }
  return {
    id: payload.id,
    sessionId: payload.sessionId ?? payload.session_id,
    exerciseId: payload.exerciseId ?? payload.exercise_id,
    overallRating: payload.overallRating ?? payload.overall_rating ?? 0,
    summaryText: payload.summaryText ?? payload.summary_text ?? '',
    errorsCount: payload.errorsCount ?? payload.errors_count ?? 0,
    topErrors: payload.topErrors ?? payload.top_errors ?? [],
    errorBreakdown: payload.errorBreakdown ?? payload.error_breakdown ?? {},
    actionItems: payload.actionItems ?? payload.action_items ?? [],
    confidenceOverall: payload.confidenceOverall ?? payload.confidence_overall ?? null,
    signalsUsed: payload.signalsUsed ?? payload.signals_used ?? {},
    version: payload.version,
    generatedAt: payload.generatedAt ?? payload.generated_at,
    updatedAt: payload.updatedAt ?? payload.updated_at,
  };
}

function toRepStatsShape(payload = {}) {
  return {
    repCount: payload.repCount ?? payload.rep_count ?? 0,
    avgRangeOfMotion: payload.avgRangeOfMotion ?? payload.avg_range_of_motion ?? null,
    avgRepDurationMs: payload.avgRepDurationMs ?? payload.avg_rep_duration_ms ?? null,
    avgFormScore: payload.avgFormScore ?? payload.avg_form_score ?? null,
    depthQualityCounts: payload.depthQualityCounts ?? payload.depth_quality_counts ?? {},
  };
}

function toReportSessionShape(payload) {
  return {
    sessionId: payload.sessionId ?? payload.session_id,
    exerciseId: payload.exerciseId ?? payload.exercise_id,
    exerciseName: payload.exerciseName ?? payload.exercise_name ?? 'Workout',
    difficulty: payload.difficulty ?? '',
    targetSets: payload.targetSets ?? payload.target_sets ?? 1,
    targetReps: payload.targetReps ?? payload.target_reps ?? 1,
    totalReps: payload.totalReps ?? payload.total_reps ?? 0,
    avgFormScore: payload.avgFormScore ?? payload.avg_form_score ?? null,
    durationSeconds: payload.durationSeconds ?? payload.duration_seconds ?? null,
    startedAt: payload.startedAt ?? payload.started_at ?? null,
    endedAt: payload.endedAt ?? payload.ended_at ?? null,
    externalLoadKg: payload.externalLoadKg ?? payload.external_load_kg ?? null,
    bodyWeightKg: payload.bodyWeightKg ?? payload.body_weight_kg ?? null,
    usesExternalLoad: payload.usesExternalLoad ?? payload.uses_external_load ?? false,
    estimatedVolumeLoadKg: payload.estimatedVolumeLoadKg ?? payload.estimated_volume_load_kg ?? null,
    feedback: toFeedbackShape(payload.feedback),
    repStats: toRepStatsShape(payload.repStats ?? payload.rep_stats),
  };
}

function toDailyReportShape(payload) {
  const summary = payload.summary ?? {};
  const user = payload.user ?? {};
  return {
    generatedAt: payload.generatedAt ?? payload.generated_at,
    user: {
      id: user.id,
      email: user.email ?? '',
      displayName: user.displayName ?? user.display_name ?? 'Athlete',
    },
    summary: {
      date: summary.date,
      timezone: summary.timezone ?? 'UTC',
      totalSessions: summary.totalSessions ?? summary.total_sessions ?? 0,
      totalReps: summary.totalReps ?? summary.total_reps ?? 0,
      totalDurationSeconds: summary.totalDurationSeconds ?? summary.total_duration_seconds ?? 0,
      avgFormScore: summary.avgFormScore ?? summary.avg_form_score ?? null,
      totalExternalLoadVolumeKg:
        summary.totalExternalLoadVolumeKg ?? summary.total_external_load_volume_kg ?? null,
    },
    sessions: Array.isArray(payload.sessions)
      ? payload.sessions.map(toReportSessionShape)
      : [],
  };
}

export async function getDailyReport({ date } = {}) {
  const query = date ? `?date=${encodeURIComponent(date)}` : '';
  const payload = await apiRequest(`/reports/daily${query}`, { auth: true });
  return toDailyReportShape(payload);
}
