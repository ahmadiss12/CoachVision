function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatDate(value) {
  if (!value) {
    return 'Today';
  }
  const date = new Date(`${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}

function formatTime(value) {
  if (!value) {
    return '--';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return '--';
  }
  return date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatDuration(seconds) {
  const safeSeconds = Number(seconds ?? 0);
  if (!Number.isFinite(safeSeconds) || safeSeconds <= 0) {
    return '0s';
  }
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = Math.round(safeSeconds % 60);
  if (minutes <= 0) {
    return `${remainder}s`;
  }
  return `${minutes}m ${remainder}s`;
}

function formatNumber(value, suffix = '') {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return '--';
  }
  const rounded = Math.round(number * 10) / 10;
  return `${rounded}${suffix}`;
}

function statCard(label, value) {
  return `
    <div class="stat">
      <div class="stat-label">${escapeHtml(label)}</div>
      <div class="stat-value">${escapeHtml(value)}</div>
    </div>
  `;
}

function feedbackBlock(feedback) {
  if (!feedback) {
    return '<p class="muted">No generated feedback is available for this workout yet.</p>';
  }

  const topErrors = feedback.topErrors?.length
    ? feedback.topErrors
        .map(
          (item) => `
            <li>
              <strong>${escapeHtml(item.label ?? item.code ?? 'Form issue')}</strong>
              <span>${escapeHtml(item.severity ?? 'medium')} severity, ${escapeHtml(item.count ?? 0)} time(s)</span>
            </li>
          `,
        )
        .join('')
    : '<li><strong>No major form issues detected</strong><span>Keep the same control and range.</span></li>';

  const actions = feedback.actionItems?.length
    ? feedback.actionItems
        .map(
          (item) => `
            <li>
              <strong>${escapeHtml(item.title ?? 'Correction')}</strong>
              <span>${escapeHtml(item.cue ?? item.howToFix ?? '')}</span>
            </li>
          `,
        )
        .join('')
    : '<li><strong>Maintain consistency</strong><span>Repeat the same tempo and body position next session.</span></li>';

  return `
    <p>${escapeHtml(feedback.summaryText)}</p>
    <div class="two-col">
      <div>
        <h4>Top feedback</h4>
        <ul>${topErrors}</ul>
      </div>
      <div>
        <h4>Coach cues</h4>
        <ul>${actions}</ul>
      </div>
    </div>
  `;
}

function workoutSection(session, index) {
  const loadText = session.usesExternalLoad
    ? `${formatNumber(session.externalLoadKg, ' kg')} carried`
    : 'Bodyweight';
  const volumeText = session.estimatedVolumeLoadKg
    ? `${formatNumber(session.estimatedVolumeLoadKg, ' kg')} volume`
    : '--';
  const repStats = session.repStats ?? {};

  return `
    <section class="workout">
      <div class="workout-head">
        <div>
          <div class="section-kicker">Workout ${index + 1}</div>
          <h3>${escapeHtml(session.exerciseName)}</h3>
          <p class="muted">${escapeHtml(session.difficulty)} &middot; ${formatTime(session.startedAt)} to ${formatTime(session.endedAt)}</p>
        </div>
        <div class="score">${escapeHtml(formatNumber(session.avgFormScore ?? session.feedback?.overallRating, '/100'))}</div>
      </div>

      <div class="stats-grid small">
        ${statCard('Target', `${session.targetSets ?? 1} x ${session.targetReps ?? 1}`)}
        ${statCard('Completed', session.totalReps ?? 0)}
        ${statCard('Duration', formatDuration(session.durationSeconds))}
        ${statCard('Weight', loadText)}
        ${statCard('Volume load', volumeText)}
        ${statCard('Rep ROM', formatNumber(repStats.avgRangeOfMotion, ' deg'))}
      </div>

      ${feedbackBlock(session.feedback)}
    </section>
  `;
}

export function buildDailyReportHtml(report) {
  const summary = report.summary ?? {};
  const sessions = report.sessions ?? [];
  const sessionSections = sessions.length
    ? sessions.map(workoutSection).join('')
    : '<section class="workout"><p class="muted">No completed workouts were found for this day.</p></section>';

  return `
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>CoachVision Daily Report</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 32px;
      color: #17201b;
      background: #f7f8f3;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 13px;
      line-height: 1.45;
    }
    .page {
      max-width: 780px;
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #d8dfd7;
      border-radius: 8px;
      padding: 28px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      border-bottom: 2px solid #20392d;
      padding-bottom: 18px;
      margin-bottom: 18px;
    }
    .brand {
      font-size: 12px;
      font-weight: 800;
      color: #2f7d59;
      text-transform: uppercase;
    }
    h1, h2, h3, h4, p { margin: 0; }
    h1 { font-size: 28px; line-height: 1.1; margin-top: 6px; }
    h2 { font-size: 18px; margin: 24px 0 12px; }
    h3 { font-size: 19px; margin-top: 4px; }
    h4 { font-size: 13px; margin: 14px 0 8px; color: #20392d; }
    .muted { color: #637268; }
    .meta {
      text-align: right;
      min-width: 190px;
      color: #435046;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin: 16px 0;
    }
    .stats-grid.small {
      grid-template-columns: repeat(3, 1fr);
    }
    .stat {
      border: 1px solid #d8dfd7;
      border-radius: 8px;
      padding: 10px;
      background: #f7faf6;
      min-height: 64px;
    }
    .stat-label {
      font-size: 10px;
      color: #637268;
      font-weight: 800;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .stat-value {
      font-size: 17px;
      color: #17201b;
      font-weight: 800;
    }
    .workout {
      border-top: 1px solid #d8dfd7;
      padding-top: 18px;
      margin-top: 18px;
      page-break-inside: avoid;
    }
    .workout-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
    }
    .section-kicker {
      font-size: 11px;
      color: #2f7d59;
      font-weight: 800;
      text-transform: uppercase;
    }
    .score {
      min-width: 76px;
      border-radius: 8px;
      background: #20392d;
      color: #ffffff;
      padding: 10px;
      text-align: center;
      font-weight: 800;
    }
    .two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-top: 8px;
    }
    ul {
      margin: 0;
      padding-left: 18px;
    }
    li {
      margin-bottom: 8px;
    }
    li strong {
      display: block;
      color: #17201b;
    }
    li span {
      color: #637268;
    }
    .notes {
      border: 1px dashed #aebaae;
      border-radius: 8px;
      padding: 14px;
      min-height: 92px;
      margin-top: 12px;
      color: #637268;
    }
    .footer {
      margin-top: 24px;
      padding-top: 12px;
      border-top: 1px solid #d8dfd7;
      color: #637268;
      font-size: 11px;
    }
  </style>
</head>
<body>
  <main class="page">
    <header class="header">
      <div>
        <div class="brand">CoachVision Daily Report</div>
        <h1>${escapeHtml(formatDate(summary.date))}</h1>
        <p class="muted">${escapeHtml(report.user?.displayName ?? 'Athlete')} &middot; ${escapeHtml(report.user?.email ?? '')}</p>
      </div>
      <div class="meta">
        <p><strong>Generated</strong></p>
        <p>${escapeHtml(new Date(report.generatedAt || Date.now()).toLocaleString())}</p>
        <p>${escapeHtml(summary.timezone ?? 'UTC')}</p>
      </div>
    </header>

    <section>
      <h2>Daily summary</h2>
      <div class="stats-grid">
        ${statCard('Workouts', summary.totalSessions ?? 0)}
        ${statCard('Total reps/sec', summary.totalReps ?? 0)}
        ${statCard('Training time', formatDuration(summary.totalDurationSeconds))}
        ${statCard('Average score', formatNumber(summary.avgFormScore, '/100'))}
      </div>
      <div class="stats-grid small">
        ${statCard('Weighted volume', summary.totalExternalLoadVolumeKg ? formatNumber(summary.totalExternalLoadVolumeKg, ' kg') : '--')}
        ${statCard('Report type', 'Daily')}
        ${statCard('Sessions included', 'Same-day completed only')}
      </div>
    </section>

    <section>
      <h2>Workout feedback</h2>
      ${sessionSections}
    </section>

    <section>
      <h2>Coach notes</h2>
      <div class="notes">Notes, corrections, or next-session focus.</div>
    </section>

    <footer class="footer">
      CoachVision feedback is AI-assisted and should be reviewed by a qualified coach for training decisions.
    </footer>
  </main>
</body>
</html>
`;
}
