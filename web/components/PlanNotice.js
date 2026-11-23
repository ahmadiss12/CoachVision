import Link from 'next/link';
import { planLabel } from '@/lib/plans';

/** Says what a plan would add in this spot and links to the plans page. */
export function PlanNotice({ plan, title, children }) {
  return (
    <div className="plan-notice">
      <span className="plan-notice-icon" aria-hidden="true">🔒</span>
      <div className="plan-notice-copy">
        <b>{title ?? `Included with ${planLabel(plan)}`}</b>
        {children ? <span className="muted">{children}</span> : null}
      </div>
      <Link href="/plans" className="plan-notice-link">
        See plans →
      </Link>
    </div>
  );
}

/** Small badge naming the plan a feature belongs to. */
export function PlanTag({ plan, note }) {
  if (!plan) return null;
  return (
    <span className="badge plan-tag" title={note}>
      {planLabel(plan)}
      {note ? <span className="plan-tag-note">· {note}</span> : null}
    </span>
  );
}
