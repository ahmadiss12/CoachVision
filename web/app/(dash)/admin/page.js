'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/auth-context';

const ROLE_TONES = { client: 'brand', trainer: 'success', admin: 'warning' };

export default function AdminPage() {
  const { user: me } = useAuth();
  const [stats, setStats] = useState(null);
  const [users, setUsers] = useState([]);
  const [query, setQuery] = useState('');
  const [error, setError] = useState(null);

  const refresh = useCallback(() => {
    api('/admin/stats')
      .then(setStats)
      .catch(() => setStats(null));
    api('/admin/users')
      .then(setUsers)
      .catch((err) => setError(err?.message || 'Failed to load users.'));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return users;
    return users.filter(
      (u) =>
        u.email.toLowerCase().includes(needle) ||
        u.displayName.toLowerCase().includes(needle) ||
        u.role.includes(needle),
    );
  }, [users, query]);

  const changeRole = async (target, role) => {
    setError(null);
    try {
      const updated = await api(`/admin/users/${target.id}/role`, { method: 'PATCH', body: { role } });
      setUsers((prev) => prev.map((u) => (u.id === updated.id ? updated : u)));
      api('/admin/stats').then(setStats).catch(() => undefined);
    } catch (err) {
      setError(err?.message || 'Role change failed.');
    }
  };

  const removeUser = async (target) => {
    if (
      !window.confirm(
        `Delete ${target.displayName} (${target.email})?\n\nThis permanently removes the account and ALL its data. This cannot be undone.`,
      )
    )
      return;
    setError(null);
    try {
      await api(`/admin/users/${target.id}`, { method: 'DELETE' });
      setUsers((prev) => prev.filter((u) => u.id !== target.id));
      api('/admin/stats').then(setStats).catch(() => undefined);
    } catch (err) {
      setError(err?.message || 'Delete failed.');
    }
  };

  return (
    <>
      <h1 className="page-title">Admin</h1>
      <p className="page-subtitle">Platform overview and user management.</p>
      {error ? <div className="error-box">{error}</div> : null}

      {stats ? (
        <div className="grid cols-4" style={{ marginBottom: 14 }}>
          <div className="stat-card">
            <div className="value">{stats.totalUsers}</div>
            <div className="label">Users</div>
          </div>
          <div className="stat-card">
            <div className="value">
              {stats.clients} / {stats.trainers}
            </div>
            <div className="label">Clients / trainers</div>
          </div>
          <div className="stat-card">
            <div className="value">{stats.completedSessions}</div>
            <div className="label">Workouts (all time)</div>
          </div>
          <div className="stat-card">
            <div className="value">{stats.sessionsLast7d}</div>
            <div className="label">Workouts (7 days)</div>
          </div>
        </div>
      ) : null}

      {stats ? (
        <div className="grid cols-2" style={{ marginBottom: 14 }}>
          <div className="stat-card">
            <div className="value">{stats.activeCoachingLinks}</div>
            <div className="label">Active coaching links</div>
          </div>
          <div className="stat-card">
            <div className="value">{stats.pendingInvites}</div>
            <div className="label">Pending invites</div>
          </div>
        </div>
      ) : null}

      <div className="card" style={{ padding: 0 }}>
        <div style={{ padding: 14, borderBottom: '1px solid var(--border)' }}>
          <input
            className="input"
            placeholder="Search by name, email, or role…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <table>
          <thead>
            <tr>
              <th>User</th>
              <th>Email</th>
              <th>Joined</th>
              <th>Role</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {visible.map((u) => (
              <tr key={u.id}>
                <td style={{ fontWeight: 800, textTransform: 'capitalize' }}>
                  {u.displayName}
                  {me?.id === u.id ? <span className="muted"> (you)</span> : null}
                </td>
                <td className="muted">{u.email}</td>
                <td className="muted">{new Date(u.createdAt).toLocaleDateString()}</td>
                <td>
                  {me?.id === u.id ? (
                    <span className={`badge ${ROLE_TONES[u.role]}`}>{u.role}</span>
                  ) : (
                    <select
                      className="select"
                      style={{ width: 120, padding: '5px 8px', fontSize: 13 }}
                      value={u.role}
                      onChange={(e) => changeRole(u, e.target.value)}
                    >
                      <option value="client">client</option>
                      <option value="trainer">trainer</option>
                      <option value="admin">admin</option>
                    </select>
                  )}
                </td>
                <td style={{ textAlign: 'right' }}>
                  {me?.id !== u.id ? (
                    <button className="btn danger small" onClick={() => removeUser(u)}>
                      Delete
                    </button>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
