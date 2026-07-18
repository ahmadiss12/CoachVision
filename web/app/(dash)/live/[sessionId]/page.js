'use client';

import { useParams, useRouter, useSearchParams } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import { buildWsUrl } from '@/lib/api';

const FORM_TONES = {
  Correct: 'var(--success)',
};

function formatClock(totalSeconds) {
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

export default function LiveSessionPage() {
  const { sessionId } = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const clientName = searchParams.get('client') || 'Client';
  const exerciseHint = searchParams.get('exercise') || '';

  const [status, setStatus] = useState('connecting'); // connecting | live | ended | error
  const [errorText, setErrorText] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const startedRef = useRef(Date.now());
  const socketRef = useRef(null);

  useEffect(() => {
    if (!sessionId) return undefined;

    const socket = new WebSocket(buildWsUrl(`/ws/observe/${sessionId}`));
    socketRef.current = socket;

    socket.onmessage = (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      if (message.type === 'observing') {
        setStatus('live');
        if (typeof message.startedAt === 'number') {
          startedRef.current = message.startedAt * 1000;
        }
      } else if (message.type === 'metrics') {
        setMetrics(message);
      } else if (message.type === 'ended') {
        setStatus('ended');
        socket.close();
      } else if (message.type === 'error') {
        setStatus('error');
        setErrorText(message.message || 'Connection rejected.');
        socket.close();
      }
    };
    socket.onerror = () => {
      setStatus((prev) => (prev === 'live' || prev === 'ended' ? prev : 'error'));
      setErrorText((prev) => prev || 'Could not connect to the live session.');
    };
    socket.onclose = () => {
      setStatus((prev) => (prev === 'ended' || prev === 'error' ? prev : 'ended'));
    };

    const keepAlive = setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send('ping');
      }
    }, 25000);

    return () => {
      clearInterval(keepAlive);
      socket.close();
    };
  }, [sessionId]);

  useEffect(() => {
    if (status !== 'live') return undefined;
    const tick = setInterval(() => {
      setElapsed(Math.max(0, Math.floor((Date.now() - startedRef.current) / 1000)));
    }, 1000);
    return () => clearInterval(tick);
  }, [status]);

  const measurementLabel = metrics?.measurementLabel === 'SEC' ? 'seconds' : 'reps';
  const formName = metrics?.formName || '—';
  const formColor = FORM_TONES[formName] || (metrics?.formName ? 'var(--warning)' : 'var(--text-dim)');

  return (
    <>
      <div className="row between">
        <div>
          <h1 className="page-title" style={{ textTransform: 'capitalize' }}>
            {clientName} — live workout
          </h1>
          <p className="page-subtitle" style={{ textTransform: 'capitalize' }}>
            {(metrics?.sessionId ? exerciseHint : exerciseHint).replace(/_/g, ' ') || 'Live session'}
          </p>
        </div>
        <span
          className={`badge ${status === 'live' ? 'danger' : status === 'ended' ? 'dim' : 'warning'}`}
          style={{ height: 'fit-content' }}
        >
          {status === 'live' ? '● LIVE' : status === 'ended' ? 'ended' : status}
        </span>
      </div>

      {status === 'error' ? (
        <div className="error-box">{errorText}</div>
      ) : null}

      {status === 'ended' ? (
        <div className="card" style={{ marginBottom: 14 }}>
          <p>
            The workout has ended.{' '}
            <a
              style={{ color: 'var(--brand)', fontWeight: 800, cursor: 'pointer' }}
              onClick={() => router.push('/clients')}
            >
              Back to clients →
            </a>{' '}
            <span className="muted">The full session breakdown will appear on the client page.</span>
          </p>
        </div>
      ) : null}

      <div className="grid cols-2" style={{ marginBottom: 14 }}>
        <div className="card" style={{ textAlign: 'center', padding: '34px 18px' }}>
          <div style={{ fontSize: 84, fontWeight: 900, lineHeight: 1, color: 'var(--brand)' }}>
            {metrics?.count ?? 0}
          </div>
          <div className="muted" style={{ marginTop: 8, fontWeight: 800, textTransform: 'uppercase', fontSize: 12 }}>
            {measurementLabel}
          </div>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: '34px 18px' }}>
          <div style={{ fontSize: 84, fontWeight: 900, lineHeight: 1 }}>{formatClock(elapsed)}</div>
          <div className="muted" style={{ marginTop: 8, fontWeight: 800, textTransform: 'uppercase', fontSize: 12 }}>
            elapsed
          </div>
        </div>
      </div>

      <div className="grid cols-2" style={{ marginBottom: 14 }}>
        <div className="stat-card">
          <div className="value" style={{ color: formColor }}>{formName}</div>
          <div className="label">
            Form
            {metrics?.formConfidence != null ? ` · ${Math.round(metrics.formConfidence * 100)}% confidence` : ''}
            {metrics?.formSource ? ` · ${metrics.formSource === 'xgboost' ? 'AI model' : 'rules'}` : ''}
          </div>
        </div>
        <div className="stat-card">
          <div className="value" style={{ textTransform: 'capitalize' }}>{metrics?.state || '—'}</div>
          <div className="label">Movement phase</div>
        </div>
      </div>

      <div className="card">
        <div className="section-title">Live coaching feedback</div>
        <p style={{ fontSize: 18, lineHeight: 1.6, fontWeight: 700 }}>
          {metrics?.feedback || (status === 'live' ? 'Waiting for movement…' : 'Waiting for connection…')}
        </p>
        {metrics?.voice ? (
          <p className="muted" style={{ marginTop: 6 }}>
            Voice cue sent to client: &ldquo;{metrics.voice}&rdquo;
          </p>
        ) : null}
      </div>
    </>
  );
}
