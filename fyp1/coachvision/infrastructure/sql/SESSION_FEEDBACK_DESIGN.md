# Session Feedback Schema Design

## Goal

Persist one human-readable feedback summary after a workout so the app can show:

- what went wrong,
- why it happened,
- how to fix it next set.

**Step 1 (design)** — agreed fields and JSON shapes below.  
**Step 2 (persistence)** — `session_feedback` is in `infrastructure/sql/database_schema_v1.sql` and `SessionFeedback` in `db/models.py` (unique `session_id`).  
**Step 3–4** — Rule-based generator `services/session_feedback_generator.py` runs when a live workout finalizes (`ws_persistence.finalize_completed_workout`). **`GET /v1/sessions/{session_id}/feedback`** returns stored feedback (`SessionFeedbackResponse` in `api/schemas.py`). **`POST …/end`** calls `ensure_session_feedback_for_completed_workout`: skips if feedback already exists (WS path); otherwise builds from **`rep_events`** when present (`signals_used.source`: `rep_events_v1`) or a minimal recap (`rest_completed_v1`).

---

## Granularity decision

Use **one summary per session** for v1.

- Reason: simplest for mobile UX ("session recap"), avoids large data volume.
- Future extension: add optional per-set summaries (`set_number`) in v2.

---

## Proposed entity: `session_feedback`

Primary key and links:

- `id` (UUID or BIGSERIAL)
- `session_id` (FK -> `sessions.id`, unique for v1)
- `user_id` (FK -> `users.id`)
- `exercise_id` (FK -> `exercises.id`)

Core summary fields:

- `overall_rating` (0-100)
- `summary_text` (short paragraph for user)
- `errors_count` (total issue instances detected)
- `top_errors` (JSON array of top issue keys by frequency/severity)
- `action_items` (JSON array of prioritized fix instructions)

Explainability fields:

- `error_breakdown` (JSON object keyed by issue type)
- `confidence_overall` (0.0-1.0; optional reliability of summary)
- `signals_used` (JSON object: source metrics used, e.g. reps analyzed)

Audit fields:

- `version` (e.g. `feedback_rule_v1`)
- `generated_at`
- `updated_at`

---

## JSON structure proposal

### `top_errors` (array)

Each item:

- `code` (e.g. `knee_valgus`)
- `label` (user-facing name)
- `count`
- `severity` (`low|medium|high`)

### `error_breakdown` (object)

Example key -> value:

- `knee_valgus`:
  - `count`
  - `severity`
  - `why`
  - `fix`
  - `evidence` (optional metrics snippets)

### `action_items` (array)

Each item:

- `priority` (1 highest)
- `title`
- `why`
- `how_to_fix`
- `cue` (short coaching phrase for UI/voice)

---

## Example record (logical)

- `session_id`: `uuid`
- `overall_rating`: `74`
- `summary_text`: `"Good consistency overall. Main issue was knee tracking on descent."`
- `errors_count`: `11`
- `top_errors`:
  - `knee_valgus` (high)
  - `shallow_depth` (medium)
- `action_items`:
  1. "Push knees out in descent"
  2. "Control descent tempo to reach target depth"

---

## Data sources for generation (next steps)

When implemented, generator should read from:

- `rep_events`
- `form_events`
- optional live labels (if persisted later)

---

## Acceptance criteria for design step

- v1 summary scope is clearly session-level.
- field list supports "what / why / fix" output.
- JSON payload shape is defined for frontend consumption.
- versioning field is included for future rule/ML evolution.

