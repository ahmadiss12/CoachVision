'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { api } from '@/lib/api';

export default function ClientsPage() {
  const router = useRouter();
  const [clients, setClients] = useState([]);
  const [liveSessions, setLiveSessions] = useState([]);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    api('/trainer/clients')
      .then(setClients)
      .catch((err) => setError(err?.message || 'Failed to load clients.'))
      .finally(() => setIsLoading(false));
  }, []);

  // Poll for live workouts so the LIVE badge appears within seconds.
  useEffect(() => {
    let cancelled = false;
    const poll = () =>
      api('/trainer/clients/live')
        .then((rows) => {
          if (!cancelled) setLiveSessions(rows);
        })
        .catch(() => undefined);
    poll();
    const interval = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const liveByClient = new Map(liveSessions.map((row) => [row.clientId, row]));

  const openClient = (client) => {
    const live = liveByClient.get(client.clientId);
    if (live) {
      router.push(
        `/live/${live.sessionId}?client=${encodeURIComponent(client.displayName)}&exercise=${encodeURIComponent(live.exerciseId)}`,
      );
    } else {
      router.push(`/clients/${client.clientId}?name=${encodeURIComponent(client.displayName)}`);
    }
  };

  return (
    <>
      <h1 className="page-title">My clients</h1>
      <p className="page-subtitle">
        {clients.length} active coaching {clients.length === 1 ? 'relationship' : 'relationships'}
        {liveSessions.length > 0 ? ` · ${liveSessions.length} training right now` : ''}.
      </p>
      {error ? <div className="error-box">{error}</div> : null}
      <div className="card" style={{ padding: 0 }}>
        {isLoading ? (
          <p className="empty">Loading…</p>
        ) : clients.length === 0 ? (
          <p className="empty">
            No clients yet.
            <br />
            Create an invite code on the Invites page and share it with your client.
          </p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Client</th>
                <th>Email</th>
                <th>Linked since</th>
                <th>Status</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {clients.map((client) => {
                const live = liveByClient.get(client.clientId);
                return (
                  <tr key={client.clientId} className="clickable" onClick={() => openClient(client)}>
                    <td style={{ fontWeight: 800, textTransform: 'capitalize' }}>{client.displayName}</td>
                    <td className="muted">{client.email}</td>
                    <td className="muted">{new Date(client.linkedAt).toLocaleDateString()}</td>
                    <td>
                      {live ? (
                        <span className="badge danger" style={{ textTransform: 'capitalize' }}>
                          ● LIVE — {String(live.exerciseId).replace(/_/g, ' ')}
                        </span>
                      ) : (
                        <span className="badge dim">offline</span>
                      )}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <span className="badge brand">{live ? 'watch →' : 'view →'}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
