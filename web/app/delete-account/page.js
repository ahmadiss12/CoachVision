import Link from 'next/link';

// Published publicly in the Play Store listing; deletion requests arrive here.
const CONTACT_EMAIL = 'ahmaisma555@gmail.com';
const LAST_UPDATED = '28 July 2026';
const APP_NAME = 'CoachVision';
const PACKAGE_NAME = 'com.ahmadiss12.coachvision';

export const metadata = {
  title: 'Delete your account | CoachVision',
  description:
    'How to permanently delete your CoachVision account and all associated data.',
};

export default function DeleteAccountPage() {
  return (
    <div className="legal-wrap">
      <Link href="/privacy" className="logo">
        Coach<span>Vision</span>
      </Link>

      <h1>Delete your account</h1>
      <p className="updated">Last updated: {LAST_UPDATED}</p>

      <p>
        This page explains how to permanently delete your {APP_NAME} account
        (<code>{PACKAGE_NAME}</code>) and everything associated with it.
      </p>

      <div className="callout warn">
        <strong>Deletion is immediate and permanent.</strong> Your workout history
        cannot be recovered afterwards, and support cannot restore it. If you only want
        to stop using the app for a while, sign out instead.
      </div>

      <h2>Option 1 — Delete from inside the app (fastest)</h2>
      <ol className="steps">
        <li>Open {APP_NAME} and sign in.</li>
        <li>
          Go to the <strong>Settings</strong> tab.
        </li>
        <li>
          Scroll to <strong>Danger zone</strong> and tap{' '}
          <strong>Delete my account</strong>.
        </li>
        <li>Enter your current password to confirm.</li>
        <li>
          Tap <strong>Delete forever</strong>. Your account is erased right away and you
          are returned to the sign-in screen.
        </li>
      </ol>

      <h2>Option 2 — Request deletion by email</h2>
      <p>
        If you cannot access the app or have forgotten your password, email{' '}
        <a href={`mailto:${CONTACT_EMAIL}?subject=Delete%20my%20CoachVision%20account`}>
          {CONTACT_EMAIL}
        </a>{' '}
        from the address registered to your account, with the subject{' '}
        <em>&ldquo;Delete my CoachVision account&rdquo;</em>.
      </p>
      <p>
        We will verify that the request comes from the registered email address and
        complete the deletion within 30 days. We may contact you to confirm ownership
        before deleting anything.
      </p>

      <h2>What gets deleted</h2>
      <p>All of the following are permanently removed:</p>
      <ul>
        <li>Your account and login credentials</li>
        <li>
          Profile details — display name, date of birth, height, weight, body-fat
          percentage, and profile photo
        </li>
        <li>Body measurement history</li>
        <li>
          Every workout session, including repetitions, hold times, form scores, and
          per-repetition joint angles
        </li>
        <li>Post-workout feedback and coaching recommendations</li>
        <li>Readiness and fatigue predictions</li>
        <li>Goals and training preferences</li>
        <li>Coaching links with any trainer, and messages you exchanged with them</li>
        <li>Programs you created, if you are a trainer</li>
      </ul>
      <p>
        After deletion your email address is released, so you can register a fresh
        account with the same address later if you wish. That new account starts empty;
        it does not restore anything.
      </p>

      <h2>What is retained, and for how long</h2>
      <ul>
        <li>
          <strong>Routine backups</strong> — copies taken before your deletion may
          persist briefly in our hosting provider&apos;s standard backup rotation before
          being overwritten. They are not used for any other purpose.
        </li>
        <li>
          <strong>Your clients&apos; own records</strong> — if you are a trainer, your
          clients&apos; workout history belongs to them and stays in their accounts.
          Your coaching link and any programs you assigned are removed.
        </li>
      </ul>
      <p>
        Video from your camera is never uploaded or stored at any point, so there is
        nothing of that kind to delete. See the{' '}
        <Link href="/privacy">privacy policy</Link> for details.
      </p>

      <h2>Just want to remove the app?</h2>
      <p>
        Uninstalling {APP_NAME} does <strong>not</strong> delete your account — your data
        stays on our servers and you can sign in again later. Use one of the options
        above if you want your data erased.
      </p>

      <h2>Questions</h2>
      <p>
        Contact <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>.
      </p>

      <div className="foot">
        <Link href="/privacy">Privacy policy</Link>
      </div>
    </div>
  );
}
