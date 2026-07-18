'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect } from 'react';
import { useAuth } from '@/lib/auth-context';

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
  const { user, isLoading, logout } = useAuth();

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
          <span className="badge brand" style={{ marginTop: 6 }}>
            {user.role}
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
