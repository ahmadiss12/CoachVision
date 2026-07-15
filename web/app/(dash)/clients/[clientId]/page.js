'use client';

import { useParams, useSearchParams } from 'next/navigation';
import { useEffect, useMemo, useState } from 'react';
import { api } from '@/lib/api';
import { Sparkline } from '@/components/Sparkline';

function formatDuration(seconds) {
  if (!seconds) return '0s';
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
}

export default function ClientDetailPage() {
  const { clientId } = useParams();
  const searchParams = useSearchParams();
  const clientName = searchParams.get('name') || 'Client';

  const [sessions, setSessions] = useState([]);
  const [assignments, setAssignments] = useState([]);
  const [detail, setDetail] = useState(null);
  const [detailFor, setDetailFor] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!clientId) return;
    api(`/trainer/clients/${clientId}/sessions`)
      .then(setSessions)
      .catch((err) => setError(err?.message || 'Failed to load sessions.'));
    api(`/trainer/clients/${clientId}/assignments`)
      .then(setAssignments)
      .catch(() => setAssignments([]));
  }, [clientId]);

  const activeAssignment = assignments.find((a) => a.status === 'active') || null;

  // Chronological series for the charts (API returns newest first).
  const chartData = useMemo(() => {
    const chrono = [...sessions].reverse();
    return {
      scores: chrono.map((s) => (s.avgFormScore != null ? Number(s.avgFormScore) : null)).filter((v) => v != null),
      reps: chrono.map((s) => s.totalReps ?? 0),
    };
  }, [sessions]);

  const openDetail = async (session) => {
    if (detailFor === session.id) {
      setDetailFor(null);
      setDetail(null);
      return;
    }
    setDetailFor(session.id);
    setDetail(null);
    try {
      setDetail(await api(`/trainer/clients/${clientId}/sessions/${session.id}`));
    } catch (err) {
      setDetail({ error: err?.message || 'Failed to load breakdown.' });
    }
  };

  const totalReps = sessions.reduce((sum, s) => sum + (s.totalReps ?? 0), 0);
  const avgScore = chartData.scores.length
    ? Math.round(chartData.scores.reduce((a, b) => a + b, 0) / chartData.scores.length)
    : null;

  return (
    <>
      <h1 className="page-title" style={{ textTransform: 'capitalize' }}>
        {clientName}
      </h1>
      <p className="page-subtitle">Session history, movement quality, and plan compliance.</p>
      {error ? <div className="error-box">{error}</div> : null}

      <div className="grid cols-4" style={{ marginBottom: 14 }}>
        <div className="stat-card">
          <div className="value">{sessions.length}</div>
          <div className="label">Completed sessions</div>
        </div>
        <div className="stat-card">
          <div className="value">{totalReps}</div>
          <div className="label">Total reps / seconds</div>
        </div>
        <div className="stat-card">
          <div className="value">{avgScore ?? '--'}</div>
          <div className="label">Avg form score</div>
        </div>
        <div className="stat-card">
          <div className="value">{activeAssignment ? activeAssignment.sessionsCompleted : '--'}</div>
          <div className="label">Plan workouts done</div>
        </div>
      </div>

      {activeAssignment ? (
        <div className="card" style={{ marginBottom: 14 }}>
          <div className="row between">
            <div>
              <div className="section-title" style={{ marginBottom: 2 }}>
                On plan: {activeAssignment.programName}
              </div>
              <span className="muted">
                Day {activeAssignment.daysElapsed ?? '--'} · started{' '}
                {new Date(activeAssignment.startDate).toLocaleDateString()} ·{' '}
                {activeAssignment.sessionsCompleted} plan workout
                {activeAssignment.sessionsCompleted === 1 ? '' : 's'} completed
              </span>
            </div>
            <span className="badge success">active</span>
          </div>
        </div>
      ) : (
        <div className="card" style={{ marginBottom: 14 }}>
          <span className="muted">No active program — assign one from the Programs page.</span>
        </div>
      )}

      <div className="grid cols-2" style={{ marginBottom: 14 }}>
        <div className="card">
          <div className="section-title">Form score over time</div>
          <Sparkline points={chartData.scores} color="var(--success)" label="per completed session" />
        </div>
        <div className="card">
          <div className="section-title">Output per session</div>
          <Sparkline points={chartData.reps} color="var(--brand)" label="reps / hold seconds" />
        </div>
      </div>

      <div className="card" style={{ padding: 0 }}>
        {sessions.length === 0 ? (
          <p className="empty">No completed workouts yet.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Exercise</th>
                <th>Date</th>
                <th>Output</th>
                <th>Duration</th>
                <th>Score</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {sessions.map((session) => (
                <SessionRow
                  key={session.id}
                  session={session}
                  isOpen={detailFor === session.id}
                  detail={detailFor === session.id ? detail : null}
                  onToggle={() => openDetail(session)}
                />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}

function SessionRow({ session, isOpen, detail, onToggle }) {
  return (
    <>
      <tr className="clickable" onClick={onToggle}>
        <td style={{ fontWeight: 800, textTransform: 'capitalize' }}>
          {String(session.exerciseId || '').replace(/_/g, ' ')}
        </td>
        <td className="muted">{session.endedAt ? new Date(session.endedAt).toLocaleString() : '--'}</td>
        <td>{session.totalReps ?? 0}</td>
        <td className="muted">{formatDuration(session.durationSeconds)}</td>
        <td>
          {session.avgFormScore != null ? (
            <span className={`badge ${session.avgFormScore >= 80 ? 'success' : session.avgFormScore >= 60 ? 'warning' : 'danger'}`}>
              {Math.round(session.avgFormScore)}
            </span>
          ) : (
            <span className="badge dim">--</span>
          )}
        </td>
        <td style={{ textAlign: 'right' }}>
          <span className="badge brand">{isOpen ? 'close' : 'details'}</span>
        </td>
      </tr>
      {isOpen ? (
        <tr>
          <td colSpan={6} style={{ background: 'var(--surface-alt)' }}>
            {!detail ? (
              <span className="muted">Loading breakdown…</span>
            ) : detail.error ? (
              <span style={{ color: 'var(--danger)' }}>{detail.error}</span>
            ) : (
              <SessionDetail detail={detail} />
            )}
          </td>
        </tr>
      ) : null}
    </>
  );
}

function SessionDetail({ detail }) {
  const feedback = detail.feedback;
  const repStats = detail.repStats;
  const targetTotal = (detail.targetSets || 1) * (detail.targetReps || 1);
  const compliance = targetTotal > 0 ? Math.round(((detail.totalReps || 0) / targetTotal) * 100) : null;

  return (
    <div style={{ display: 'grid', gap: 12, padding: '6px 0' }}>
      <div className="row" style={{ flexWrap: 'wrap', gap: 18 }}>
        <span className="muted">
          Target <b style={{ color: 'var(--text)' }}>{detail.targetSets}×{detail.targetReps}</b>
        </span>
        <span className="muted">
          Done <b style={{ color: 'var(--text)' }}>{detail.totalReps ?? 0}</b>
        </span>
        {compliance != null ? (
          <span className={`badge ${compliance >= 90 ? 'success' : compliance >= 60 ? 'warning' : 'danger'}`}>
            {compliance}% compliance
          </span>
        ) : null}
        {detail.usesExternalLoad ? (
          <span className="muted">
            Load <b style={{ color: 'var(--text)' }}>{detail.externalLoadKg}kg</b>
            {detail.estimatedVolumeLoadKg != null ? ` · volume ${detail.estimatedVolumeLoadKg}kg` : ''}
          </span>
        ) : null}
        {repStats?.repCount > 0 ? (
          <span className="muted">
            ROM <b style={{ color: 'var(--text)' }}>{repStats.avgRangeOfMotion ?? '--'}°</b> · tempo{' '}
            <b style={{ color: 'var(--text)' }}>
              {repStats.avgRepDurationMs != null ? `${(repStats.avgRepDurationMs / 1000).toFixed(1)}s` : '--'}
            </b>
          </span>
        ) : null}
      </div>

      {feedback ? (
        <>
          <div>
            <span className="badge brand">rating {feedback.overallRating}/100</span>
            <p style={{ marginTop: 8, lineHeight: 1.6 }}>{feedback.summaryText}</p>
          </div>
          {feedback.topErrors?.length ? (
            <div>
              <div className="section-title" style={{ fontSize: 13 }}>
                Errors detected
              </div>
              <div className="row" style={{ flexWrap: 'wrap', gap: 8 }}>
                {feedback.topErrors.map((err) => (
                  <span
                    key={err.code}
                    className={`badge ${err.severity === 'high' ? 'danger' : err.severity === 'low' ? 'success' : 'warning'}`}
                  >
                    {err.label} ×{err.count}
                  </span>
                ))}
              </div>
            </div>
          ) : null}
          {feedback.actionItems?.length ? (
            <div>
              <div className="section-title" style={{ fontSize: 13 }}>
                Coaching actions
              </div>
              <ul style={{ paddingLeft: 18, display: 'grid', gap: 6 }}>
                {feedback.actionItems.map((action) => (
                  <li key={`${action.priority}-${action.title}`} style={{ lineHeight: 1.6 }}>
                    <b>{action.title}</b> — {action.howToFix}
                    {action.cue ? <i style={{ color: 'var(--text-dim)' }}> (&ldquo;{action.cue}&rdquo;)</i> : null}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </>
      ) : (
        <span className="muted">No form review was generated for this session.</span>
      )}
    </div>
  );
}
