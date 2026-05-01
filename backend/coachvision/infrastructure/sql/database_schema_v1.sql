-- Option C database foundation (PostgreSQL)
-- Scope: users, exercises, sessions, live events, fatigue/readiness, gamification
-- Notes:
--   1) This is an initial schema script to bootstrap the backend database.
--   2) Exercise IDs are aligned with dispatcher exercise values.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'difficulty_level') THEN
        CREATE TYPE difficulty_level AS ENUM ('beginner', 'intermediate', 'advanced');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'session_status') THEN
        CREATE TYPE session_status AS ENUM ('planned', 'active', 'completed', 'aborted');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'fatigue_level') THEN
        CREATE TYPE fatigue_level AS ENUM ('low', 'moderate', 'high');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'depth_quality') THEN
        CREATE TYPE depth_quality AS ENUM ('good', 'shallow', 'very_shallow');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'event_severity') THEN
        CREATE TYPE event_severity AS ENUM ('info', 'low', 'medium', 'high');
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL,
    avatar_url TEXT,
    timezone TEXT DEFAULT 'UTC',
    locale TEXT DEFAULT 'en',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS exercises (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    default_difficulty difficulty_level NOT NULL DEFAULT 'intermediate',
    target_joints TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    exercise_id TEXT NOT NULL REFERENCES exercises(id),
    difficulty difficulty_level NOT NULL,
    status session_status NOT NULL DEFAULT 'planned',
    target_sets INT NOT NULL DEFAULT 1 CHECK (target_sets > 0),
    target_reps INT NOT NULL DEFAULT 1 CHECK (target_reps > 0),
    planned_rest_seconds INT NOT NULL DEFAULT 60 CHECK (planned_rest_seconds >= 0),
    total_reps INT NOT NULL DEFAULT 0 CHECK (total_reps >= 0),
    avg_form_score NUMERIC(5,2) CHECK (avg_form_score >= 0 AND avg_form_score <= 100),
    duration_seconds INT CHECK (duration_seconds >= 0),
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT sessions_time_order_chk
        CHECK (ended_at IS NULL OR started_at IS NULL OR ended_at >= started_at)
);

CREATE TABLE IF NOT EXISTS session_sets (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    set_number INT NOT NULL CHECK (set_number > 0),
    target_reps INT CHECK (target_reps > 0),
    completed_reps INT NOT NULL DEFAULT 0 CHECK (completed_reps >= 0),
    avg_form_score NUMERIC(5,2) CHECK (avg_form_score >= 0 AND avg_form_score <= 100),
    duration_seconds INT CHECK (duration_seconds >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, set_number)
);

CREATE TABLE IF NOT EXISTS rep_events (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    set_id BIGINT REFERENCES session_sets(id) ON DELETE SET NULL,
    rep_number INT NOT NULL CHECK (rep_number > 0),
    min_angle NUMERIC(6,2),
    max_angle NUMERIC(6,2),
    range_of_motion NUMERIC(6,2),
    duration_ms INT CHECK (duration_ms >= 0),
    depth_quality depth_quality,
    form_score NUMERIC(5,2) CHECK (form_score >= 0 AND form_score <= 100),
    extra JSONB NOT NULL DEFAULT '{}'::JSONB,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS form_events (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    set_id BIGINT REFERENCES session_sets(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    severity event_severity NOT NULL DEFAULT 'info',
    message TEXT NOT NULL,
    confidence NUMERIC(5,4) CHECK (confidence >= 0 AND confidence <= 1),
    extra JSONB NOT NULL DEFAULT '{}'::JSONB,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fatigue_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    exercise_id TEXT NOT NULL REFERENCES exercises(id),
    readiness_score INT NOT NULL CHECK (readiness_score >= 0 AND readiness_score <= 100),
    fatigue_level fatigue_level NOT NULL,
    recommendation TEXT NOT NULL,
    factors JSONB NOT NULL DEFAULT '[]'::JSONB,
    model_version TEXT NOT NULL DEFAULT 'rule_v1',
    input_snapshot JSONB NOT NULL DEFAULT '{}'::JSONB,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS readiness_features_daily (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    exercise_id TEXT NOT NULL REFERENCES exercises(id),
    feature_date DATE NOT NULL,
    sessions_7d INT NOT NULL DEFAULT 0 CHECK (sessions_7d >= 0),
    reps_7d INT NOT NULL DEFAULT 0 CHECK (reps_7d >= 0),
    avg_form_score_7d NUMERIC(5,2) CHECK (avg_form_score_7d >= 0 AND avg_form_score_7d <= 100),
    avg_rom_7d NUMERIC(7,3),
    rom_trend_7d NUMERIC(7,3),
    avg_rep_duration_ms_7d NUMERIC(10,2),
    days_since_last_session INT CHECK (days_since_last_session >= 0),
    sleep_hours NUMERIC(4,2),
    soreness_score INT CHECK (soreness_score >= 1 AND soreness_score <= 5),
    stress_score INT CHECK (stress_score >= 1 AND stress_score <= 5),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, exercise_id, feature_date)
);

CREATE TABLE IF NOT EXISTS achievements (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    xp_reward INT NOT NULL DEFAULT 0 CHECK (xp_reward >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_achievements (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    achievement_id TEXT NOT NULL REFERENCES achievements(id),
    unlocked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, achievement_id)
);

CREATE TABLE IF NOT EXISTS streaks (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    current_streak_days INT NOT NULL DEFAULT 0 CHECK (current_streak_days >= 0),
    longest_streak_days INT NOT NULL DEFAULT 0 CHECK (longest_streak_days >= 0),
    last_activity_date DATE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS xp_events (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    source TEXT NOT NULL,
    points INT NOT NULL CHECK (points >= 0),
    meta JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_exercise_started
    ON sessions (user_id, exercise_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_status_user
    ON sessions (status, user_id);

CREATE INDEX IF NOT EXISTS idx_rep_events_session_ts
    ON rep_events (session_id, ts);

CREATE INDEX IF NOT EXISTS idx_form_events_session_ts
    ON form_events (session_id, ts);

CREATE INDEX IF NOT EXISTS idx_fatigue_predictions_user_exercise_generated
    ON fatigue_predictions (user_id, exercise_id, generated_at DESC);

CREATE INDEX IF NOT EXISTS idx_readiness_features_user_exercise_date
    ON readiness_features_daily (user_id, exercise_id, feature_date DESC);

CREATE INDEX IF NOT EXISTS idx_xp_events_user_created
    ON xp_events (user_id, created_at DESC);

INSERT INTO exercises (id, name, description, default_difficulty, target_joints)
VALUES
    ('squat', 'Squat', 'Lower-body compound squat movement.', 'intermediate', ARRAY['hip', 'knee', 'ankle']),
    ('pushup', 'Push-Up', 'Upper-body pushing movement.', 'intermediate', ARRAY['shoulder', 'elbow', 'core']),
    ('lunge', 'Lunge', 'Single-leg lower-body movement.', 'intermediate', ARRAY['hip', 'knee', 'ankle']),
    ('deadlift', 'Deadlift', 'Hip-hinge posterior chain movement.', 'intermediate', ARRAY['hip', 'knee', 'back']),
    ('plank', 'Plank', 'Core isometric hold exercise.', 'intermediate', ARRAY['core', 'shoulder']),
    ('bicep_curl', 'Bicep Curl', 'Elbow flexion upper-body movement.', 'intermediate', ARRAY['elbow', 'shoulder']),
    ('shoulder_press', 'Shoulder Press', 'Overhead pressing movement.', 'intermediate', ARRAY['shoulder', 'elbow']),
    ('situp', 'Sit-Up', 'Trunk flexion core exercise.', 'intermediate', ARRAY['core', 'hip']),
    ('jumping_jack', 'Jumping Jack', 'Full-body cardio movement.', 'beginner', ARRAY['shoulder', 'hip', 'ankle']),
    ('high_knees', 'High Knees', 'Cardio drill emphasizing hip flexion.', 'beginner', ARRAY['hip', 'knee']),
    ('mountain_climber', 'Mountain Climber', 'Dynamic core and cardio movement.', 'intermediate', ARRAY['core', 'hip', 'shoulder']),
    ('wall_sit', 'Wall Sit', 'Isometric lower-body hold.', 'beginner', ARRAY['hip', 'knee'])
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS fatigue_calibration_events (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    exercise_id TEXT NOT NULL REFERENCES exercises(id),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    prediction_id UUID REFERENCES fatigue_predictions(id) ON DELETE SET NULL,
    predicted_readiness_score INT,
    predicted_fatigue_level TEXT,
    actual_performance_score NUMERIC(5,2) NOT NULL,
    readiness_delta INT NOT NULL,
    performance_drop_observed BOOLEAN NOT NULL DEFAULT FALSE,
    detail JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fatigue_calibration_user_exercise
    ON fatigue_calibration_events (user_id, exercise_id, created_at DESC);

-- Post-session human-readable feedback summary (one row per session, v1)
CREATE TABLE IF NOT EXISTS session_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    exercise_id TEXT NOT NULL REFERENCES exercises(id),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    overall_rating INT NOT NULL DEFAULT 0 CHECK (overall_rating >= 0 AND overall_rating <= 100),
    summary_text TEXT NOT NULL DEFAULT '',
    errors_count INT NOT NULL DEFAULT 0 CHECK (errors_count >= 0),
    top_errors JSONB NOT NULL DEFAULT '[]'::JSONB,
    error_breakdown JSONB NOT NULL DEFAULT '{}'::JSONB,
    action_items JSONB NOT NULL DEFAULT '[]'::JSONB,
    confidence_overall NUMERIC(4,3) CHECK (confidence_overall IS NULL OR (confidence_overall >= 0 AND confidence_overall <= 1)),
    signals_used JSONB NOT NULL DEFAULT '{}'::JSONB,
    version TEXT NOT NULL DEFAULT 'feedback_rule_v1',
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id)
);

CREATE INDEX IF NOT EXISTS idx_session_feedback_user_generated
    ON session_feedback (user_id, generated_at DESC);

-- Retention policy (execute as scheduled jobs):
-- 1) Keep hot event data for 180 days:
--      DELETE FROM rep_events  WHERE ts < NOW() - INTERVAL '180 days';
--      DELETE FROM form_events WHERE ts < NOW() - INTERVAL '180 days';
-- 2) Before deletion, optionally archive old rows to cold storage table/files.
-- 3) Keep summary tables (sessions, fatigue_predictions, readiness_features_daily) long-term.

COMMIT;
