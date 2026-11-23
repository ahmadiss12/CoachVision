'use client';

import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { api, loadTokens, login as apiLogin, logout as apiLogout } from './api';
import { normalizeEntitlements, UNRESTRICTED } from './plans';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [entitlements, setEntitlements] = useState(UNRESTRICTED);
  const [isLoading, setIsLoading] = useState(true);

  // On failure the previous value is kept (unrestricted until the first
  // answer), so a flaky connection never hides a feature the account has.
  const refreshEntitlements = useCallback(async () => {
    try {
      const next = normalizeEntitlements(await api('/me/entitlements'));
      setEntitlements(next);
      return next;
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!loadTokens()) {
        setIsLoading(false);
        return;
      }
      try {
        const me = await api('/users/me');
        if (!cancelled) setUser(me);
        await refreshEntitlements();
      } catch {
        // tokens invalid — stay signed out
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshEntitlements]);

  const login = useCallback(async (email, password) => {
    const me = await apiLogin(email, password);
    setUser(me);
    await refreshEntitlements();
    return me;
  }, [refreshEntitlements]);

  const logout = useCallback(() => {
    apiLogout();
    setUser(null);
    setEntitlements(UNRESTRICTED);
  }, []);

  return (
    <AuthContext.Provider value={{ user, entitlements, isLoading, login, logout, refreshEntitlements }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
  return ctx;
}
