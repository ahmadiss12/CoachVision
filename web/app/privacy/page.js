import Link from 'next/link';

// Published publicly in the Play Store listing; data requests arrive here.
const CONTACT_EMAIL = 'ahmaisma555@gmail.com';
const LAST_UPDATED = '28 July 2026';

export const metadata = {
  title: 'Privacy Policy | CoachVision',
  description:
    'How CoachVision collects, uses, stores, and deletes your personal and fitness data.',
};

export default function PrivacyPolicyPage() {
  return (
    <div className="legal-wrap">
      <Link href="/privacy" className="logo">
        Coach<span>Vision</span>
      </Link>

      <h1>Privacy Policy</h1>
      <p className="updated">Last updated: {LAST_UPDATED}</p>

      <p>
        CoachVision is a fitness coaching app that uses your phone camera to count
        repetitions and give feedback on exercise form. This policy explains what
        data we collect, why we collect it, who it is shared with, and how you can
        delete it.
      </p>

      <div className="callout">
        <strong>The camera feed is not uploaded.</strong> Pose detection runs entirely
        on your device. The app sends only numeric body-joint coordinates (for example,
        the angle of your knee) to our server so it can count repetitions and assess
        form. Video and images from your workout camera are never transmitted to us
        and are never stored.
      </div>

      <h2>1. Who we are</h2>
      <p>
        CoachVision is operated by an independent developer. For any privacy question
        or request, contact <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>.
      </p>

      <h2>2. Data we collect</h2>
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th>What it includes</th>
            <th>Why</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Account</td>
            <td>Email address, password, display name, account type</td>
            <td>To create and secure your account</td>
          </tr>
          <tr>
            <td>Profile</td>
            <td>
              Date of birth, height, weight, body-fat percentage, time zone, language,
              and a profile photo if you choose to upload one
            </td>
            <td>To personalise targets and calculate BMI</td>
          </tr>
          <tr>
            <td>Health &amp; fitness</td>
            <td>
              Workout sessions, exercise type, repetitions, hold durations, form
              scores, joint angles and range of motion per repetition, session
              duration, and any weight lifted
            </td>
            <td>To count reps, score form, and show your history</td>
          </tr>
          <tr>
            <td>Wellbeing inputs</td>
            <td>
              Sleep hours, soreness, and stress ratings that you enter yourself
            </td>
            <td>To estimate readiness and recommend your next session</td>
          </tr>
          <tr>
            <td>Camera-derived data</td>
            <td>
              Numeric body-joint coordinates computed on your device during a workout
            </td>
            <td>To count repetitions and detect form errors</td>
          </tr>
          <tr>
            <td>Messages</td>
            <td>
              Messages exchanged between you and your trainer, if you use the coaching
              feature
            </td>
            <td>To deliver coaching conversations</td>
          </tr>
        </tbody>
      </table>

      <p>
        We do not collect your contacts, precise location, advertising identifiers, or
        browsing activity. CoachVision contains no advertising and no third-party
        analytics or tracking SDKs.
      </p>

      <h2>3. Device permissions</h2>
      <ul>
        <li>
          <strong>Camera</strong> — required for live workout tracking. Used only while
          a workout screen is open, and processed on your device as described above.
        </li>
        <li>
          <strong>Photo library</strong> — requested only if you choose to set a profile
          picture. We access the single image you select, nothing else.
        </li>
      </ul>
      <p>You can refuse or revoke either permission in your device settings.</p>

      <h2>4. How your data is used</h2>
      <ul>
        <li>To count repetitions and give real-time form feedback during workouts.</li>
        <li>To produce post-workout summaries and track progress over time.</li>
        <li>To estimate training readiness and suggest your next session.</li>
        <li>
          To share your workout results with a trainer, but only if you have an active
          coaching link with them.
        </li>
        <li>To secure accounts and keep the service working correctly.</li>
      </ul>
      <p>
        Your data is not sold, rented, or used for advertising or profiling unrelated
        to the app&apos;s coaching features.
      </p>

      <h2>5. Who your data is shared with</h2>
      <p>
        We do not sell personal data. It is shared only with the service providers
        needed to run the app, and with your trainer if you choose to have one.
      </p>
      <table>
        <thead>
          <tr>
            <th>Recipient</th>
            <th>Purpose</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Your trainer</td>
            <td>
              If you accept a trainer&apos;s invitation, they can see your workout
              sessions, results, and messages you send them. Ending the coaching link
              stops this access.
            </td>
          </tr>
          <tr>
            <td>Render</td>
            <td>Hosts the application server and database</td>
          </tr>
          <tr>
            <td>Vercel</td>
            <td>Hosts this website and the trainer dashboard</td>
          </tr>
          <tr>
            <td>Expo (EAS)</td>
            <td>Builds the app and delivers app updates</td>
          </tr>
          <tr>
            <td>jsDelivr and Google (Cloud Storage)</td>
            <td>
              Deliver the on-device pose-detection model to your app. These providers
              receive your IP address as part of that download, in the same way any
              website request does. They do not receive your workout or account data.
            </td>
          </tr>
        </tbody>
      </table>
      <p>
        We may also disclose data where required by law, or to protect the rights and
        safety of users.
      </p>

      <h2>6. International transfers</h2>
      <p>
        Our service providers operate data centres in various countries, so your data
        may be processed outside your country of residence.
      </p>

      <h2>7. How long we keep data</h2>
      <p>
        Account, profile, and workout data is kept for as long as your account exists.
        When you delete your account, it is removed immediately and permanently — see
        the next section.
      </p>

      <h2>8. Deleting your account and data</h2>
      <div className="callout">
        <p>
          You can delete your account at any time from{' '}
          <strong>Settings &rarr; Delete my account</strong> inside the app. You will be
          asked to confirm your password.
        </p>
        <p style={{ marginBottom: 0 }}>
          Full instructions, including how to request deletion by email, are on the{' '}
          <Link href="/delete-account">account deletion page</Link>.
        </p>
      </div>
      <p>
        Deletion permanently removes your account record, profile, body measurements,
        workout and repetition history, session feedback, readiness predictions, and
        coaching links. It is immediate and cannot be undone. Messages you sent to a
        trainer are removed together with your account.
      </p>
      <p>
        Backups taken before deletion may persist for a short period in our hosting
        provider&apos;s routine backup rotation before being overwritten.
      </p>

      <h2>9. Your rights</h2>
      <p>
        Depending on where you live, you may have the right to access, correct, export,
        or delete your personal data, and to object to or restrict its processing. You
        can view and edit most of your data directly in the app, and delete all of it
        using the steps above. For any other request, contact{' '}
        <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>.
      </p>

      <h2>10. Security</h2>
      <p>
        Passwords are stored only as salted cryptographic hashes — never in a readable
        form. Traffic between the app and our server is encrypted in transit using
        HTTPS and secure WebSockets, and access to your data requires a valid signed
        authentication token. No system can be guaranteed completely secure, but we
        take reasonable measures to protect your information.
      </p>

      <h2>11. Children</h2>
      <p>
        CoachVision is not directed at children under 13, and we do not knowingly
        collect data from them. If you believe a child has provided us with personal
        data, contact us and we will delete it.
      </p>

      <h2>12. Changes to this policy</h2>
      <p>
        We may update this policy as the app evolves. The &ldquo;last updated&rdquo;
        date at the top will always reflect the current version, and significant
        changes will be highlighted in the app.
      </p>

      <h2>13. Contact</h2>
      <p>
        Questions, requests, or complaints:{' '}
        <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>
      </p>

      <div className="foot">
        <Link href="/delete-account">Account deletion</Link>
      </div>
    </div>
  );
}
