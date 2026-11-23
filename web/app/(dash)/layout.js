'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect } from 'react';
import { useAuth } from '@/lib/auth-context';
import { planLabel } from '@/lib/plans';

const TRAINER_LINKS = [
  { href: '/clients', label: '👥 Clients' },
  { href: '/programs', label: '📋 Programs' },
  { href: '/messages', label: '💬 Messages' },
  { href: '/invites', label: '✉️ Invites' },
];

const ADMIN_LINKS = [{ href: '/admin', label: '🛡️ Admin' }];

export default function DashLayout({ children }) {
  const router = useRouter();
  const pathname = usePathname();
  const { user, entitlements, isLoading, logout } = useAuth();

  useEffect(() => {
    if (isLoading) return;
    if (!user || user.role === 'client') {
      router.replace('/login');
    }
  }, [user, isLoading, router]);

  if (isLoading || !user || user.role === 'client') {
    return null;
  }

  const links = [
    ...(user.role === 'trainer' || user.role === 'admin' ? TRAINER_LINKS : []),
    ...(user.role === 'admin' ? ADMIN_LINKS : []),
  ];

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="logo">
          Coach<span>Vision</span>
        </div>
        {links.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className={`nav-link ${pathname.startsWith(link.href) ? 'active' : ''}`}
          >
            {link.label}
          </Link>
        ))}
        <div className="spacer" />
        <div className="whoami">
          <b>{user.display_name}</b>
          {user.email}
          <br />
          <span className="row" style={{ marginTop: 6, gap: 6, flexWrap: 'wrap' }}>
            <span className="badge brand">{user.role}</span>
            {user.role === 'trainer' && entitlements.isLoaded ? (
              <Link href="/plans" className="badge dim" title="See what each plan includes">
                {planLabel(entitlements.plan)} plan{entitlements.earlyAccess ? ' · early access' : ''}
              </Link>
            ) : null}
          </span>
        </div>
        <button
          className="btn secondary"
          onClick={() => {
            logout();
            router.replace('/login');
          }}
        >
          Sign out
        </button>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
