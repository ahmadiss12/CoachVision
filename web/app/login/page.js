'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { useAuth } from '@/lib/auth-context';

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [isBusy, setIsBusy] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setError(null);
    setIsBusy(true);
    try {
      const me = await login(email.trim(), password);
      if (me.role === 'client') {
        setError('This dashboard is for trainers and admins — clients use the mobile app.');
        return;
      }
      router.replace(me.role === 'admin' ? '/admin' : '/clients');
    } catch (err) {
      setError(err?.message || 'Login failed.');
    } finally {
      setIsBusy(false);
    }
  };

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <h1>
          Coach<span>Vision</span>
        </h1>
        <p className="muted">Trainer &amp; admin dashboard. Sign in with your account.</p>
        {error ? <div className="error-box">{error}</div> : null}
        <div>
          <label className="field" htmlFor="email">
            Email
          </label>
          <input
            id="email"
            className="input"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            required
          />
        </div>
        <div>
          <label className="field" htmlFor="password">
            Password
          </label>
          <input
            id="password"
            className="input"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
            required
          />
        </div>
        <button className="btn" type="submit" disabled={isBusy}>
          {isBusy ? 'Signing in…' : 'Sign in'}
        </button>
      </form>
    </div>
  );
}
