'use client';

import { useEffect } from 'react';
import { useAuth } from '@/lib/auth-context';
import { PLAN_ORDER, planLabel } from '@/lib/plans';

const ALL_EXERCISES = 12;

function count(value, unit) {
  return value == null ? 'Unlimited' : `${value} ${unit}${value === 1 ? '' : 's'}`;
}

// Every row is read from the server's plan list, so this table cannot drift
// from what the server enforces.
const ROWS = [
  { label: 'Active clients', value: (p) => count(p.limits?.maxClients, 'client') },
  { label: 'Programs', value: (p) => count(p.limits?.maxPrograms, 'program') },
  {
    label: 'Exercises in programs',
    value: (p) => (p.limits?.exerciseIds == null ? `All ${ALL_EXERCISES}` : `${p.limits.exerciseIds.length} bodyweight`),
  },
  { label: 'Weights in programs', value: (p) => (p.features?.includes('weighted_training') ? '✓' : '—') },
  {
    label: 'Client session history',
    value: (p) => (p.limits?.historyDays == null ? 'Full history' : `Last ${p.limits.historyDays} days`),
  },
  {
    label: 'Client session review',
    value: (p) => (p.features?.includes('full_client_session_detail') ? 'Full review' : 'Summary'),
  },
  { label: 'Live monitoring', value: (p) => (p.features?.includes('live_monitoring') ? '✓' : '—') },
];

export default function PlansPage() {
  const { entitlements, refreshEntitlements } = useAuth();
  const plans = entitlements.plans ?? {};
  const available = PLAN_ORDER.filter((plan) => plans[plan]);

  useEffect(() => {
    refreshEntitlements();
  }, [refreshEntitlements]);

  return (
    <>
      <h1 className="page-title">Plans</h1>
      <p className="page-subtitle">
        {entitlements.isLoaded ? `You are on ${planLabel(entitlements.plan)}.` : 'What each plan includes.'}
      </p>

      <div className="card" style={{ marginBottom: 14 }}>
        {!entitlements.isLoaded ? (
          <p className="muted">Plans are not available on this server yet.</p>
        ) : entitlements.earlyAccess ? (
          <p>
            <b>Early access:</b> <span className="muted">every feature below is free for now. You will be told before that changes.</span>
          </p>
        ) : (
          <p className="muted">Plus and Pro are not available to buy yet. They will appear here when they are.</p>
        )}
      </div>

      <div className="card" style={{ padding: 0 }}>
        {available.length === 0 ? (
          <p className="empty">Plan details are not available right now.</p>
        ) : (
          <table className="plan-table">
            <thead>
              <tr>
                <th />
                {available.map((plan) => (
                  <th key={plan} className={plan === entitlements.plan ? 'current' : ''}>
                    {planLabel(plan)}
                    {plan === entitlements.plan ? ' (yours)' : ''}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row) => (
                <tr key={row.label}>
                  <td style={{ fontWeight: 800 }}>{row.label}</td>
                  {available.map((plan) => (
                    <td key={plan}>{row.value(plans[plan])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
