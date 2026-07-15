'use client';

import { useCallback, useEffect, useState } from 'react';
import { api } from '@/lib/api';

const EXERCISE_CATALOG = [
  { id: 'squat', label: 'Squat', load: true },
  { id: 'pushup', label: 'Push-up' },
  { id: 'lunge', label: 'Lunge', load: true },
  { id: 'deadlift', label: 'Deadlift', load: true },
  { id: 'bicep_curl', label: 'Bicep curl', load: true },
  { id: 'shoulder_press', label: 'Shoulder press', load: true },
  { id: 'situp', label: 'Sit-up' },
  { id: 'jumping_jack', label: 'Jumping jack' },
  { id: 'high_knees', label: 'High knees' },
  { id: 'mountain_climber', label: 'Mountain climber' },
  { id: 'plank', label: 'Plank (sec)', hold: true },
  { id: 'wall_sit', label: 'Wall sit (sec)', hold: true },
];

export default function ProgramsPage() {
  const [programs, setPrograms] = useState([]);
  const [clients, setClients] = useState([]);
  const [error, setError] = useState(null);
  const [showBuilder, setShowBuilder] = useState(false);
  const [assigningProgram, setAssigningProgram] = useState(null);

  const refresh = useCallback(() => {
    api('/trainer/programs')
      .then(setPrograms)
      .catch((err) => setError(err?.message || 'Failed to load programs.'));
    api('/trainer/clients')
      .then(setClients)
      .catch(() => setClients([]));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const remove = async (program) => {
    if (!window.confirm(`Delete "${program.name}"? Clients assigned to it will lose their plan.`)) return;
    try {
      await api(`/trainer/programs/${program.id}`, { method: 'DELETE' });
      refresh();
    } catch (err) {
      setError(err?.message || 'Delete failed.');
    }
  };

  return (
    <>
      <div className="row between">
        <div>
          <h1 className="page-title">Programs</h1>
          <p className="page-subtitle">Build training plans and assign them to clients.</p>
        </div>
        <button className="btn" onClick={() => setShowBuilder((v) => !v)}>
          {showBuilder ? 'Close builder' : '+ New program'}
        </button>
      </div>
      {error ? <div className="error-box">{error}</div> : null}

      {showBuilder ? (
        <ProgramBuilder
          onSaved={() => {
            setShowBuilder(false);
            refresh();
          }}
        />
      ) : null}

      <div className="grid" style={{ marginTop: 14 }}>
        {programs.length === 0 ? (
          <div className="card">
            <p className="empty">No programs yet — create your first one.</p>
          </div>
        ) : (
          programs.map((program) => (
            <div key={program.id} className="card">
              <div className="row between">
                <div>
                  <div className="section-title" style={{ marginBottom: 2 }}>
                    {program.name}
                  </div>
                  <span className="muted">
                    {program.weeks} week{program.weeks > 1 ? 's' : ''} · {program.days.length} training day
                    {program.days.length === 1 ? '' : 's'} ·{' '}
                    {program.days.reduce((sum, d) => sum + d.exercises.length, 0)} exercises
                  </span>
                </div>
                <div className="row">
                  <button
                    className="btn secondary small"
                    onClick={() => setAssigningProgram(assigningProgram === program.id ? null : program.id)}
                  >
                    Assign
                  </button>
                  <button className="btn danger small" onClick={() => remove(program)}>
                    Delete
                  </button>
                </div>
              </div>

              <div className="row" style={{ flexWrap: 'wrap', gap: 8, marginTop: 12 }}>
                {program.days.map((day) => (
                  <span key={day.id} className="badge dim">
                    Day {day.dayIndex}:{' '}
                    {day.exercises
                      .map((e) => `${String(e.exerciseId).replace(/_/g, ' ')} ${e.targetSets}×${e.targetReps}`)
                      .join(', ')}
                  </span>
                ))}
              </div>

              {assigningProgram === program.id ? (
                <AssignPicker
                  program={program}
                  clients={clients}
                  onDone={() => {
                    setAssigningProgram(null);
                    refresh();
                  }}
                />
              ) : null}
            </div>
          ))
        )}
      </div>
    </>
  );
}

function AssignPicker({ program, clients, onDone }) {
  const [busyId, setBusyId] = useState(null);
  const [message, setMessage] = useState(null);

  const assign = async (client) => {
    setBusyId(client.clientId);
    setMessage(null);
    try {
      await api(`/trainer/programs/${program.id}/assign`, {
        method: 'POST',
        body: { clientId: client.clientId },
      });
      setMessage(`Assigned to ${client.displayName} — starts today.`);
      setTimeout(onDone, 1200);
    } catch (err) {
      setMessage(err?.message || 'Assign failed.');
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div style={{ marginTop: 12, borderTop: '1px solid var(--border)', paddingTop: 12 }}>
      {message ? <p className="muted" style={{ marginBottom: 8 }}>{message}</p> : null}
      {clients.length === 0 ? (
        <p className="muted">No active clients — invite one first.</p>
      ) : (
        <div className="row" style={{ flexWrap: 'wrap', gap: 8 }}>
          {clients.map((client) => (
            <button
              key={client.clientId}
              className="btn secondary small"
              disabled={busyId !== null}
              onClick={() => assign(client)}
              style={{ textTransform: 'capitalize' }}
            >
              {busyId === client.clientId ? 'Assigning…' : `→ ${client.displayName}`}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function ProgramBuilder({ onSaved }) {
  const [name, setName] = useState('');
  const [weeks, setWeeks] = useState(1);
  const [days, setDays] = useState([{ dayIndex: 1, exercises: [] }]);
  const [error, setError] = useState(null);
  const [isSaving, setIsSaving] = useState(false);

  const addDay = () => {
    const used = new Set(days.map((d) => d.dayIndex));
    let next = 1;
    while (used.has(next) && next <= weeks * 7) next += 1;
    if (next > weeks * 7) {
      setError(`A ${weeks}-week cycle has at most ${weeks * 7} days.`);
      return;
    }
    setDays((prev) => [...prev, { dayIndex: next, exercises: [] }]);
  };

  const patchExercise = (dayIndex, position, patch) => {
    setDays((prev) =>
      prev.map((d) =>
        d.dayIndex === dayIndex
          ? { ...d, exercises: d.exercises.map((e, i) => (i === position ? { ...e, ...patch } : e)) }
          : d,
      ),
    );
  };

  const save = async () => {
    setError(null);
    if (!name.trim()) {
      setError('Give the program a name.');
      return;
    }
    const trainingDays = days.filter((d) => d.exercises.length > 0);
    if (!trainingDays.length) {
      setError('Add at least one exercise to a day.');
      return;
    }
    setIsSaving(true);
    try {
      await api('/trainer/programs', {
        method: 'POST',
        body: {
          name: name.trim(),
          weeks,
          days: trainingDays.map((d) => ({
            dayIndex: d.dayIndex,
            exercises: d.exercises.map((e) => ({
              exerciseId: e.exerciseId,
              targetSets: e.targetSets,
              targetReps: e.targetReps,
              restSeconds: e.restSeconds,
              externalLoadKg: e.externalLoadKg > 0 ? e.externalLoadKg : null,
            })),
          })),
        },
      });
      onSaved();
    } catch (err) {
      setError(err?.message || 'Save failed.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="card" style={{ marginTop: 14, display: 'grid', gap: 14 }}>
      {error ? <div className="error-box" style={{ marginBottom: 0 }}>{error}</div> : null}
      <div className="grid cols-2">
        <div>
          <label className="field">Program name</label>
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Beginner full body" />
        </div>
        <div>
          <label className="field">Cycle length (weeks)</label>
          <select className="select" value={weeks} onChange={(e) => setWeeks(Number(e.target.value))}>
            {[1, 2, 3, 4].map((w) => (
              <option key={w} value={w}>
                {w} week{w > 1 ? 's' : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      {days.map((day) => (
        <div key={day.dayIndex} style={{ border: '1px solid var(--border)', borderRadius: 10, padding: 14 }}>
          <div className="row between" style={{ marginBottom: 10 }}>
            <b>Day {day.dayIndex}</b>
            <button
              className="btn danger small"
              onClick={() => setDays((prev) => prev.filter((d) => d.dayIndex !== day.dayIndex))}
            >
              Remove day
            </button>
          </div>

          {day.exercises.map((exercise, position) => {
            const meta = EXERCISE_CATALOG.find((e) => e.id === exercise.exerciseId);
            return (
              <div key={position} className="row" style={{ flexWrap: 'wrap', gap: 8, marginBottom: 8 }}>
                <span className="badge brand" style={{ minWidth: 110, justifyContent: 'center' }}>
                  {meta?.label ?? exercise.exerciseId}
                </span>
                <NumberField label="sets" value={exercise.targetSets} min={1} max={10} onChange={(v) => patchExercise(day.dayIndex, position, { targetSets: v })} />
                <NumberField label={meta?.hold ? 'sec' : 'reps'} value={exercise.targetReps} min={1} max={200} onChange={(v) => patchExercise(day.dayIndex, position, { targetReps: v })} />
                <NumberField label="rest s" value={exercise.restSeconds} min={0} max={300} onChange={(v) => patchExercise(day.dayIndex, position, { restSeconds: v })} />
                {meta?.load ? (
                  <NumberField label="kg" value={exercise.externalLoadKg} min={0} max={200} onChange={(v) => patchExercise(day.dayIndex, position, { externalLoadKg: v })} />
                ) : null}
                <button
                  className="btn danger small"
                  onClick={() =>
                    setDays((prev) =>
                      prev.map((d) =>
                        d.dayIndex === day.dayIndex
                          ? { ...d, exercises: d.exercises.filter((_, i) => i !== position) }
                          : d,
                      ),
                    )
                  }
                >
                  ✕
                </button>
              </div>
            );
          })}

          <select
            className="select"
            style={{ maxWidth: 260 }}
            value=""
            onChange={(e) => {
              const exerciseId = e.target.value;
              if (!exerciseId) return;
              setDays((prev) =>
                prev.map((d) =>
                  d.dayIndex === day.dayIndex
                    ? {
                        ...d,
                        exercises: [
                          ...d.exercises,
                          { exerciseId, targetSets: 3, targetReps: 10, restSeconds: 60, externalLoadKg: 0 },
                        ],
                      }
                    : d,
                ),
              );
            }}
          >
            <option value="">+ Add exercise…</option>
            {EXERCISE_CATALOG.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.label}
              </option>
            ))}
          </select>
        </div>
      ))}

      <div className="row">
        <button className="btn secondary" onClick={addDay}>
          + Add training day
        </button>
        <button className="btn" onClick={save} disabled={isSaving}>
          {isSaving ? 'Saving…' : 'Save program'}
        </button>
      </div>
    </div>
  );
}

function NumberField({ label, value, min, max, onChange }) {
  return (
    <label className="row" style={{ gap: 5 }}>
      <span className="muted" style={{ fontSize: 11, fontWeight: 800, textTransform: 'uppercase' }}>
        {label}
      </span>
      <input
        className="input"
        type="number"
        style={{ width: 74 }}
        value={value}
        min={min}
        max={max}
        onChange={(e) => onChange(Math.max(min, Math.min(max, Number(e.target.value) || 0)))}
      />
    </label>
  );
}
