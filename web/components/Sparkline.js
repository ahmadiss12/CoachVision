'use client';

/**
 * Dependency-free SVG line chart with dots and a soft area fill.
 * points: array of numbers (chronological). Renders nothing with < 2 points.
 */
export function Sparkline({ points, width = 520, height = 120, color = 'var(--brand)', label }) {
  const values = (points || []).filter((v) => Number.isFinite(v));
  if (values.length < 2) {
    return <p className="muted">Not enough data yet.</p>;
  }

  const pad = 8;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const stepX = (width - pad * 2) / (values.length - 1);
  const toY = (v) => height - pad - ((v - min) / span) * (height - pad * 2);

  const coords = values.map((v, i) => [pad + i * stepX, toY(v)]);
  const path = coords.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const area = `${path} L${coords[coords.length - 1][0].toFixed(1)},${height - pad} L${pad},${height - pad} Z`;

  return (
    <div>
      {label ? (
        <div className="row between" style={{ marginBottom: 6 }}>
          <span className="muted">{label}</span>
          <span className="muted">
            min {Math.round(min)} · max {Math.round(max)}
          </span>
        </div>
      ) : null}
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}>
        <path d={area} fill={color} opacity="0.12" />
        <path d={path} fill="none" stroke={color} strokeWidth="2.5" strokeLinejoin="round" />
        {coords.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="3" fill={color} />
        ))}
      </svg>
    </div>
  );
}
