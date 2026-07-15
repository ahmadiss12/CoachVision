'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { api } from '@/lib/api';

export default function ClientsPage() {
  const router = useRouter();
  const [clients, setClients] = useState([]);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    api('/trainer/clients')
      .then(setClients)
      .catch((err) => setError(err?.message || 'Failed to load clients.'))
      .finally(() => setIsLoading(false));
  }, []);

  return (
    <>
      <h1 className="page-title">My clients</h1>
      <p className="page-subtitle">
        {clients.length} active coaching {clients.length === 1 ? 'relationship' : 'relationships'}.
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
                <th />
              </tr>
            </thead>
            <tbody>
              {clients.map((client) => (
                <tr
                  key={client.clientId}
                  className="clickable"
                  onClick={() => router.push(`/clients/${client.clientId}?name=${encodeURIComponent(client.displayName)}`)}
                >
                  <td style={{ fontWeight: 800, textTransform: 'capitalize' }}>{client.displayName}</td>
                  <td className="muted">{client.email}</td>
                  <td className="muted">{new Date(client.linkedAt).toLocaleDateString()}</td>
                  <td style={{ textAlign: 'right' }}>
                    <span className="badge brand">view →</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
