'use client';

import { useCallback, useEffect, useState } from 'react';
import { api } from '@/lib/api';

const STATUS_TONES = { pending: 'warning', accepted: 'success', revoked: 'danger', expired: 'dim' };

export default function InvitesPage() {
  const [invites, setInvites] = useState([]);
  const [error, setError] = useState(null);
  const [isCreating, setIsCreating] = useState(false);
  const [copiedId, setCopiedId] = useState(null);

  const refresh = useCallback(() => {
    api('/trainer/invites')
      .then(setInvites)
      .catch((err) => setError(err?.message || 'Failed to load invites.'));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const create = async () => {
    setIsCreating(true);
    setError(null);
    try {
      await api('/trainer/invites', { method: 'POST', body: {} });
      refresh();
    } catch (err) {
      setError(err?.message || 'Could not create an invite.');
    } finally {
      setIsCreating(false);
    }
  };

  const revoke = async (invite) => {
    if (!window.confirm('Revoke this invite code? It will stop working immediately.')) return;
    try {
      await api(`/trainer/invites/${invite.id}`, { method: 'DELETE' });
      refresh();
    } catch (err) {
      setError(err?.message || 'Revoke failed.');
    }
  };

  const copy = async (invite) => {
    try {
      await navigator.clipboard.writeText(invite.token);
      setCopiedId(invite.id);
      setTimeout(() => setCopiedId(null), 1500);
    } catch {
      // clipboard unavailable — user can select manually
    }
  };

  return (
    <>
      <div className="row between">
        <div>
          <h1 className="page-title">Client invites</h1>
          <p className="page-subtitle">Codes are single-use and expire after 7 days.</p>
        </div>
        <button className="btn" onClick={create} disabled={isCreating}>
          {isCreating ? 'Creating…' : '+ New invite code'}
        </button>
      </div>
      {error ? <div className="error-box">{error}</div> : null}

      <div className="card" style={{ padding: 0 }}>
        {invites.length === 0 ? (
          <p className="empty">No invites yet — create a code and send it to your client.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Code</th>
                <th>Status</th>
                <th>Expires</th>
                <th>For</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {invites.map((invite) => (
                <tr key={invite.id}>
                  <td className="mono">{invite.token}</td>
                  <td>
                    <span className={`badge ${STATUS_TONES[invite.status] ?? 'dim'}`}>{invite.status}</span>
                  </td>
                  <td className="muted">{new Date(invite.expiresAt).toLocaleDateString()}</td>
                  <td className="muted">{invite.email || 'anyone'}</td>
                  <td style={{ textAlign: 'right' }}>
                    {invite.status === 'pending' ? (
                      <span className="row" style={{ justifyContent: 'flex-end' }}>
                        <button className="btn secondary small" onClick={() => copy(invite)}>
                          {copiedId === invite.id ? 'Copied!' : 'Copy'}
                        </button>
                        <button className="btn danger small" onClick={() => revoke(invite)}>
                          Revoke
                        </button>
                      </span>
                    ) : null}
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
