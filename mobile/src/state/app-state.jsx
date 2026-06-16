import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { getExerciseMetadata } from '../constants/exercise-metadata';
import {
  loginUser,
  logoutUser,
  refreshAuthToken,
  registerUser,
} from '../services/api/auth';
import { clearApiAuthTokens, setApiAuthTokens } from '../services/api/client';
import { listExercises } from '../services/api/exercises';
import {
  listFatigueHistory,
  predictFatigue as requestFatiguePrediction,
} from '../services/api/fatigue';
import {
  createSession,
  endSession,
  getSessionFeedback,
  listSessions,
  startSession,
} from '../services/api/sessions';
import {
  createBodyMetricEntry as requestCreateBodyMetricEntry,
  getCurrentUser,
  listBodyMetricEntries,
  updateCurrentUser,
} from '../services/api/users';
import {
  clearStoredAuthTokens,
  loadStoredAuthTokens,
  loadStoredGoals,
  loadStoredThemeMode,
  loadStoredWorkoutPreferences,
  saveStoredAuthTokens,
  saveStoredGoals,
  saveStoredThemeMode,
  saveStoredWorkoutPreferences,
  userStorageKey,
} from '../services/local/app-storage';
const defaultMetrics = {
    count: 0,
    state: 'idle',
    angle: 180,
    feedback: 'Start your set when ready.',
    progress: 0,
    formName: 'Correct',
    formConfidence: null,
    formSource: null,
    confidence: 0,
};
const defaultWorkoutPreferences = {
    difficulty: 'beginner',
};
function computeAgeFromDob(dateOfBirth) {
    if (!dateOfBirth)
        return 18;
    const dob = new Date(dateOfBirth);
    if (Number.isNaN(dob.getTime()))
        return 18;
    const today = new Date();
    let age = today.getFullYear() - dob.getFullYear();
    const monthDiff = today.getMonth() - dob.getMonth();
    const hasBirthdayPassed = monthDiff > 0 || (monthDiff === 0 && today.getDate() >= dob.getDate());
    if (!hasBirthdayPassed)
        age -= 1;
    return Math.max(1, age);
}
const AppStateContext = createContext(null);
function formatExerciseName(exerciseId) {
    const meta = getExerciseMetadata(exerciseId);
    if (meta.label) {
        return meta.label;
    }
    return String(exerciseId || 'Workout')
        .split(/[_-]+/)
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' ');
}

function summaryFromSession(session) {
    const meta = getExerciseMetadata(session.exerciseId);
    const endedAt = session.endedAt || session.createdAt || new Date().toISOString();
    const startedAt = session.startedAt || session.createdAt || endedAt;
    const startedMs = new Date(startedAt).getTime();
    const endedMs = new Date(endedAt).getTime();
    const fallbackDuration = Number.isFinite(startedMs) && Number.isFinite(endedMs)
        ? Math.max(1, Math.round((endedMs - startedMs) / 1000))
        : 0;
    return {
        id: session.id,
        sessionId: session.id,
        exerciseId: session.exerciseId,
        measurementType: meta.measurementType,
        metricLabel: meta.metricLabel,
        startedAt,
        endedAt,
        durationSec: session.durationSeconds ?? fallbackDuration,
        exerciseName: formatExerciseName(session.exerciseId),
        difficulty: session.difficulty,
        targetSets: session.targetSets ?? 1,
        targetReps: session.targetReps ?? 1,
        externalLoadKg: session.externalLoadKg ?? 0,
        bodyWeightKg: session.bodyWeightKg ?? null,
        reps: session.totalReps ?? 0,
        score: session.avgFormScore ?? 0,
        notes: '',
        feedback: null,
        fatiguePrediction: null,
    };
}

function normalizeBodyMetricEntry(item) {
    return {
        id: String(item.id ?? `${item.date ?? item.entry_date}-${Date.now()}`),
        date: item.date ?? item.entry_date ?? new Date().toISOString().slice(0, 10),
        weightKg: Number(item.weightKg ?? item.weight_kg ?? 0),
        bodyFatPercent: Number(item.bodyFatPercent ?? item.body_fat_percent ?? 0),
    };
}

export function AppStateProvider({ children }) {
    const [authUser, setAuthUser] = useState(null);
    const [authTokens, setAuthTokens] = useState(null);
    const [profile, setProfile] = useState(null);
    const [goals, setGoals] = useState(null);
    const [themeMode, setThemeMode] = useState('dark');
    const [workoutPreferences, setWorkoutPreferences] = useState(defaultWorkoutPreferences);
    const [isHydrating, setIsHydrating] = useState(true);
    const [isBusy, setIsBusy] = useState(false);
    const [latestError, setLatestError] = useState(null);
    const [currentSession, setCurrentSession] = useState(null);
    const [latestSummary, setLatestSummary] = useState(null);
    const [latestFeedback, setLatestFeedback] = useState(null);
    const [latestFatiguePrediction, setLatestFatiguePrediction] = useState(null);
    const [fatigueHistory, setFatigueHistory] = useState([]);
    const [availableExercises, setAvailableExercises] = useState([]);
    const [history, setHistory] = useState([]);
    const [bodyMetrics, setBodyMetrics] = useState([]);
    const userKeyRef = useRef(null);

    const setTokensEverywhere = (tokens) => {
        setAuthTokens(tokens);
        if (tokens) {
            setApiAuthTokens(tokens);
            saveStoredAuthTokens(tokens).catch(() => undefined);
        }
        else {
            clearApiAuthTokens();
            clearStoredAuthTokens().catch(() => undefined);
        }
    };

    const resetUserScopedState = () => {
        userKeyRef.current = null;
        setProfile(null);
        setGoals(null);
        setWorkoutPreferences(defaultWorkoutPreferences);
        setBodyMetrics([]);
        setCurrentSession(null);
        setLatestSummary(null);
        setLatestFeedback(null);
        setLatestFatiguePrediction(null);
        setFatigueHistory([]);
        setHistory([]);
    };

    const ensureFreshAccessToken = async () => {
        if (!authTokens?.refreshToken) {
            return false;
        }
        try {
            const next = await refreshAuthToken(authTokens.refreshToken);
            setTokensEverywhere(next);
            return true;
        } catch {
            setTokensEverywhere(null);
            setAuthUser(null);
            return false;
        }
    };

    const hydrateProfileFromUser = (user) => {
        if (!user?.date_of_birth || !user?.height_cm || !user?.weight_kg) {
            return;
        }
        const safeDob = user.date_of_birth;
        const safeHeightCm = Number(user.height_cm);
        const safeWeightKg = Number(user.weight_kg);
        const safeBodyFatPercent = Number(user.body_fat_percent ?? 0);
        const heightM = safeHeightCm / 100;
        const bmi = safeWeightKg / (heightM * heightM);
        setProfile({
            dateOfBirth: safeDob,
            age: computeAgeFromDob(safeDob),
            heightCm: safeHeightCm,
            weightKg: safeWeightKg,
            bodyFatPercent: safeBodyFatPercent,
            bmi: Number.isFinite(bmi) ? Number(bmi.toFixed(1)) : 0,
            avatarUri: user.avatar_url ?? undefined,
        });
    };

    const loadUserLocalState = async (user) => {
        const key = userStorageKey(user);
        userKeyRef.current = key;
        const [storedGoals, storedWorkoutPreferences] = await Promise.all([
            loadStoredGoals(key),
            loadStoredWorkoutPreferences(key),
        ]);
        setGoals(storedGoals ?? null);
        setWorkoutPreferences({
            ...defaultWorkoutPreferences,
            ...(storedWorkoutPreferences ?? {}),
        });
    };

    const applyAuthenticatedUser = async (user) => {
        setAuthUser(user);
        hydrateProfileFromUser(user);
        await loadUserLocalState(user);
    };

    const loadCurrentUser = async () => {
        try {
            const user = await getCurrentUser();
            await applyAuthenticatedUser(user);
            setLatestError(null);
            return user;
        } catch (error) {
            if (error?.status === 401) {
                const refreshed = await ensureFreshAccessToken();
                if (refreshed) {
                    const user = await getCurrentUser();
                    await applyAuthenticatedUser(user);
                    setLatestError(null);
                    return user;
                }
            }
            throw error;
        }
    };

    const loadHistory = async () => {
        try {
            const sessions = await listSessions();
            const summaries = sessions.map(summaryFromSession);
            setHistory(summaries);
            setLatestError(null);
            return summaries;
        } catch (error) {
            if (error?.status === 401) {
                const refreshed = await ensureFreshAccessToken();
                if (refreshed) {
                    const sessions = await listSessions();
                    const summaries = sessions.map(summaryFromSession);
                    setHistory(summaries);
                    setLatestError(null);
                    return summaries;
                }
            }
            setLatestError(error?.message || 'Failed to load workout history.');
            return [];
        }
    };

    const loadBodyMetrics = async () => {
        try {
            const items = await listBodyMetricEntries();
            const metrics = items.map(normalizeBodyMetricEntry);
            setBodyMetrics(metrics);
            return metrics;
        } catch (error) {
            if (error?.status === 401) {
                const refreshed = await ensureFreshAccessToken();
                if (refreshed) {
                    const items = await listBodyMetricEntries();
                    const metrics = items.map(normalizeBodyMetricEntry);
                    setBodyMetrics(metrics);
                    return metrics;
                }
            }
            return [];
        }
    };

    useEffect(() => {
        let cancelled = false;

        const hydrateStoredState = async () => {
            try {
                const [storedThemeMode, storedTokens] = await Promise.all([
                    loadStoredThemeMode(),
                    loadStoredAuthTokens(),
                ]);

                if (cancelled) {
                    return;
                }

                if (storedThemeMode) {
                    setThemeMode(storedThemeMode);
                }

                if (!storedTokens?.refreshToken) {
                    return;
                }

                setAuthTokens(storedTokens);
                setApiAuthTokens(storedTokens);

                let user = null;
                try {
                    user = await getCurrentUser();
                } catch (error) {
                    if (error?.status !== 401) {
                        throw error;
                    }
                    const refreshedTokens = await refreshAuthToken(storedTokens.refreshToken);
                    if (cancelled) {
                        return;
                    }
                    setTokensEverywhere(refreshedTokens);
                    user = await getCurrentUser();
                }

                if (cancelled) {
                    return;
                }

                await applyAuthenticatedUser(user);
                await Promise.all([loadHistory(), loadBodyMetrics()]);
                setLatestError(null);
            } catch {
                if (!cancelled) {
                    setTokensEverywhere(null);
                    setAuthUser(null);
                    resetUserScopedState();
                }
            } finally {
                if (!cancelled) {
                    setIsHydrating(false);
                }
            }
        };

        hydrateStoredState();

        return () => {
            cancelled = true;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const login = async (input, optionalPassword) => {
        const email = typeof input === 'string' ? input : input?.email;
        const password = typeof input === 'string' ? optionalPassword : input?.password;
        if (!email || !password) {
            setLatestError('Email and password are required.');
            return null;
        }
        setIsBusy(true);
        try {
            const tokens = await loginUser({ email, password });
            setTokensEverywhere(tokens);
            resetUserScopedState();
            const user = await loadCurrentUser();
            await loadHistory();
            await loadBodyMetrics();
            return user;
        } catch (error) {
            setLatestError(error?.message || 'Login failed. Please try again.');
            return null;
        } finally {
            setIsBusy(false);
        }
    };

    const register = async (input, optionalPassword) => {
        const email = typeof input === 'string' ? input : input?.email;
        const password = typeof input === 'string' ? optionalPassword : input?.password;
        if (!email || !password) {
            setLatestError('Email and password are required.');
            return null;
        }
        setIsBusy(true);
        try {
            const inferredName = email.split('@')[0] || 'CoachVision User';
            const tokens = await registerUser({
                email,
                password,
                displayName: inferredName,
            });
            setTokensEverywhere(tokens);
            resetUserScopedState();
            const user = await loadCurrentUser();
            await loadHistory();
            await loadBodyMetrics();
            return user;
        } catch (error) {
            setLatestError(error?.message || 'Registration failed. Please try again.');
            return null;
        } finally {
            setIsBusy(false);
        }
    };

    const logout = async () => {
        try {
            if (authTokens?.accessToken) {
                await logoutUser();
            }
        } catch {
            // Logout is best-effort on client side.
        }
        setTokensEverywhere(null);
        setAuthUser(null);
        resetUserScopedState();
    };
    const saveProfile = (payload) => {
        const safeDob = payload.dateOfBirth || '2000-01-01';
        const safeAge = computeAgeFromDob(safeDob);
        const safeHeightCm = payload.heightCm > 0 ? payload.heightCm : 170;
        const safeWeightKg = payload.weightKg > 0 ? payload.weightKg : 70;
        const safeBodyFatPercent = payload.bodyFatPercent >= 0 ? payload.bodyFatPercent : 20;
        const heightM = safeHeightCm / 100;
        const bmi = safeWeightKg / (heightM * heightM);
        const nextProfile = {
            dateOfBirth: safeDob,
            age: safeAge,
            heightCm: safeHeightCm,
            weightKg: safeWeightKg,
            bodyFatPercent: safeBodyFatPercent,
            bmi: Number.isFinite(bmi) ? Number(bmi.toFixed(1)) : 0,
            avatarUri: payload.avatarUri,
        };
        setProfile(nextProfile);
        if (authUser) {
            const nameFromEmail = authUser.email?.split('@')[0] || 'CoachVision User';
            updateCurrentUser({
                display_name: nameFromEmail,
                avatar_url: payload.avatarUri ?? null,
                date_of_birth: safeDob,
                height_cm: safeHeightCm,
                weight_kg: safeWeightKg,
                body_fat_percent: safeBodyFatPercent,
            }).catch(() => undefined);
        }
        setBodyMetrics((prev) => {
            if (prev.length > 0)
                return prev;
            const initialMetric = {
                id: `${Date.now()}`,
                date: new Date().toISOString().slice(0, 10),
                weightKg: safeWeightKg,
                bodyFatPercent: safeBodyFatPercent,
            };
            requestCreateBodyMetricEntry({
                date: initialMetric.date,
                weightKg: safeWeightKg,
                bodyFatPercent: safeBodyFatPercent,
            }).catch(() => undefined);
            return [initialMetric];
        });
        return nextProfile;
    };
    const updateProfileAvatar = (avatarUri) => {
        setProfile((prev) => {
            if (!prev)
                return prev;
            return { ...prev, avatarUri };
        });
    };
    const saveGoals = (payload) => {
        const next = {
            targetWeightKg: payload.targetWeightKg && payload.targetWeightKg > 0
                ? payload.targetWeightKg
                : profile?.weightKg ?? 68,
            targetBodyFatPercent: payload.targetBodyFatPercent !== undefined && payload.targetBodyFatPercent >= 0
                ? payload.targetBodyFatPercent
                : profile?.bodyFatPercent ?? 18,
            weeklyWorkoutTarget: payload.weeklyWorkoutTarget && payload.weeklyWorkoutTarget > 0 ? payload.weeklyWorkoutTarget : 4,
        };
        setGoals(next);
        saveStoredGoals(userKeyRef.current, next).catch(() => undefined);
        return next;
    };
    const addBodyMetricEntry = (payload) => {
        const safeWeight = payload.weightKg > 0 ? payload.weightKg : profile?.weightKg ?? 70;
        const safeBodyFat = payload.bodyFatPercent >= 0 ? payload.bodyFatPercent : profile?.bodyFatPercent ?? 20;
        const date = payload.date || new Date().toISOString().slice(0, 10);
        const tempId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        const optimisticEntry = {
            id: tempId,
            date,
            weightKg: safeWeight,
            bodyFatPercent: safeBodyFat,
        };
        setBodyMetrics((prev) => [
            optimisticEntry,
            ...prev,
        ]);
        setProfile((prev) => {
            if (!prev)
                return prev;
            const heightM = prev.heightCm / 100;
            const bmi = safeWeight / (heightM * heightM);
            return {
                ...prev,
                weightKg: safeWeight,
                bodyFatPercent: safeBodyFat,
                bmi: Number.isFinite(bmi) ? Number(bmi.toFixed(1)) : prev.bmi,
            };
        });
        requestCreateBodyMetricEntry({
            date,
            weightKg: safeWeight,
            bodyFatPercent: safeBodyFat,
        })
            .then((saved) => {
                const savedEntry = normalizeBodyMetricEntry(saved);
                setBodyMetrics((prev) => prev.map((item) => (item.id === tempId ? savedEntry : item)));
            })
            .catch((error) => {
                setLatestError(error?.message || 'Body entry saved locally but could not sync.');
            });
    };
    const clearError = () => setLatestError(null);
    const toggleThemeMode = () => {
        setThemeMode((prev) => {
            const next = prev === 'dark' ? 'light' : 'dark';
            saveStoredThemeMode(next).catch(() => undefined);
            return next;
        });
    };
    const saveWorkoutPreferences = (payload) => {
        const next = {
            ...workoutPreferences,
            ...(payload ?? {}),
        };
        if (!['beginner', 'intermediate', 'advanced'].includes(next.difficulty)) {
            next.difficulty = defaultWorkoutPreferences.difficulty;
        }
        setWorkoutPreferences(next);
        saveStoredWorkoutPreferences(userKeyRef.current, next).catch(() => undefined);
        return next;
    };
    const predictFatigue = async ({
        exerciseId,
        userContext = {},
        recentWindowDays = 14,
        silent = false,
    } = {}) => {
        if (!exerciseId) {
            if (!silent) {
                setLatestError('Please choose an exercise first.');
            }
            return null;
        }
        if (!authTokens?.accessToken) {
            if (!silent) {
                setLatestError('Please sign in first.');
            }
            return null;
        }

        const run = () => requestFatiguePrediction({ exerciseId, userContext, recentWindowDays });
        try {
            const prediction = await run();
            setLatestFatiguePrediction(prediction);
            if (!silent) {
                setLatestError(null);
            }
            return prediction;
        } catch (error) {
            let requestError = error;
            if (error?.status === 401) {
                const refreshed = await ensureFreshAccessToken();
                if (refreshed) {
                    try {
                        const prediction = await run();
                        setLatestFatiguePrediction(prediction);
                        if (!silent) {
                            setLatestError(null);
                        }
                        return prediction;
                    } catch (retryError) {
                        requestError = retryError;
                    }
                }
            }
            if (!silent) {
                setLatestError(requestError?.message || 'Unable to check fatigue readiness.');
            }
            return null;
        }
    };
    const loadFatigueHistory = async (exerciseId) => {
        if (!exerciseId || !authTokens?.accessToken) {
            return [];
        }
        try {
            const items = await listFatigueHistory(exerciseId);
            setFatigueHistory(items);
            return items;
        } catch (error) {
            if (error?.status === 401) {
                const refreshed = await ensureFreshAccessToken();
                if (refreshed) {
                    const items = await listFatigueHistory(exerciseId);
                    setFatigueHistory(items);
                    return items;
                }
            }
            return [];
        }
    };
    const loadExercises = async () => {
        try {
            const items = await listExercises();
            setAvailableExercises(items);
            return items;
        } catch (error) {
            setLatestError(error?.message || 'Failed to load exercises.');
            return [];
        }
    };

    const startWorkout = async (config) => {
        if (!config?.exerciseName) {
            setLatestError('Please choose an exercise first.');
            return null;
        }
        if (!authTokens?.accessToken) {
            setLatestError('Please sign in first.');
            return null;
        }
        setIsBusy(true);
        try {
            const externalLoadKg = Number(config.readinessContext?.externalLoadKg ?? 0);
            const bodyWeightKg = Number(config.readinessContext?.bodyWeightKg ?? profile?.weightKg ?? 0);
            const created = await createSession({
                exerciseId: config.exerciseName,
                difficulty: config.difficulty || 'beginner',
                targetSets: config.targetSets || 1,
                targetReps: config.targetReps || 1,
                externalLoadKg: Number.isFinite(externalLoadKg) ? externalLoadKg : 0,
                bodyWeightKg: Number.isFinite(bodyWeightKg) && bodyWeightKg > 0 ? bodyWeightKg : null,
            });
            const started = await startSession(created.id);
            const nextSession = {
                config,
                sessionId: started.id,
                startedAt: started.startedAt || new Date().toISOString(),
                latestMetrics: defaultMetrics,
                fatiguePrediction: config.fatiguePrediction ?? null,
            };
            setCurrentSession(nextSession);
            setLatestError(null);
            return nextSession;
        } catch (error) {
            setLatestError(error?.message || 'Unable to start workout session.');
            return null;
        } finally {
            setIsBusy(false);
        }
    };

    const updateMetrics = (metrics) => {
        setCurrentSession((prev) => {
            if (!prev) {
                return prev;
            }
            return {
                ...prev,
                latestMetrics: metrics,
            };
        });
    };

    const finishWorkout = async ({ endedSummary = null } = {}) => {
        if (!currentSession)
            return null;
        setIsBusy(true);
        try {
            const counterSummary = endedSummary?.summary ?? {};
            const exerciseMeta = getExerciseMetadata(currentSession.config.exerciseName);
            const isHoldExercise = exerciseMeta.measurementType === 'hold';
            const fallbackReps = isHoldExercise
                ? Math.round(
                    counterSummary.total_seconds
                    ?? counterSummary.totalSeconds
                    ?? counterSummary.total_hold_time
                    ?? currentSession.latestMetrics.totalHoldTimeSec
                    ?? currentSession.latestMetrics.bestHoldSec
                    ?? currentSession.latestMetrics.holdDurationSec
                    ?? currentSession.latestMetrics.count
                    ?? 0
                )
                : counterSummary.total_reps
                    ?? counterSummary.totalReps
                    ?? endedSummary?.totalReps
                    ?? currentSession.latestMetrics.count
                    ?? 0;
            const fallbackScore = Math.min(100, Math.round((currentSession.latestMetrics.confidence || 0) * 100));
            const ended = await endSession(currentSession.sessionId, {
                totalReps: fallbackReps,
                avgFormScore: fallbackScore,
                externalLoadKg: currentSession.config.readinessContext?.externalLoadKg ?? 0,
                bodyWeightKg: currentSession.config.readinessContext?.bodyWeightKg ?? profile?.weightKg ?? null,
            });
            let feedback = null;
            try {
                feedback = await getSessionFeedback(currentSession.sessionId);
                setLatestFeedback(feedback);
            } catch {
                feedback = null;
            }

            const endedAt = ended.endedAt || new Date().toISOString();
            const startedAt = ended.startedAt || currentSession.startedAt;
            const startedMs = new Date(startedAt).getTime();
            const endedMs = new Date(endedAt).getTime();
            const durationSec = ended.durationSeconds ?? Math.max(1, Math.round((endedMs - startedMs) / 1000));
            const reps = isHoldExercise
                ? Math.round(
                    counterSummary.total_seconds
                    ?? counterSummary.totalSeconds
                    ?? counterSummary.total_hold_time
                    ?? ended.totalReps
                    ?? currentSession.latestMetrics.totalHoldTimeSec
                    ?? currentSession.latestMetrics.bestHoldSec
                    ?? currentSession.latestMetrics.holdDurationSec
                    ?? currentSession.latestMetrics.count
                    ?? 0
                )
                : counterSummary.total_reps
                    ?? counterSummary.totalReps
                    ?? endedSummary?.totalReps
                    ?? ended.totalReps
                    ?? currentSession.latestMetrics.count
                    ?? 0;
            const score = feedback?.overallRating
                ?? ended.avgFormScore
                ?? counterSummary.detection_quality
                ?? Math.min(100, Math.round((currentSession.latestMetrics.confidence || 0) * 100));
            const nextFatiguePrediction = await predictFatigue({
                exerciseId: currentSession.config.exerciseName,
                userContext: currentSession.config.readinessContext ?? {},
                recentWindowDays: 14,
                silent: true,
            });

            const summary = {
                id: ended.id,
                sessionId: ended.id,
                exerciseId: currentSession.config.exerciseName,
                measurementType: currentSession.config.measurementType ?? exerciseMeta.measurementType,
                metricLabel: currentSession.config.metricLabel ?? exerciseMeta.metricLabel,
                startedAt,
                endedAt,
                durationSec,
                exerciseName: formatExerciseName(currentSession.config.exerciseName),
                difficulty: currentSession.config.difficulty,
                targetSets: currentSession.config.targetSets ?? 1,
                targetReps: currentSession.config.targetReps ?? 1,
                externalLoadKg: ended.externalLoadKg ?? currentSession.config.readinessContext?.externalLoadKg ?? 0,
                bodyWeightKg: ended.bodyWeightKg ?? currentSession.config.readinessContext?.bodyWeightKg ?? null,
                reps,
                score,
                notes: feedback?.summaryText || currentSession.latestMetrics.feedback,
                feedback,
                fatiguePrediction: nextFatiguePrediction ?? currentSession.fatiguePrediction ?? null,
            };
            setLatestSummary(summary);
            setHistory((prev) => [summary, ...prev]);
            setCurrentSession(null);
            return summary;
        } catch (error) {
            setLatestError(error?.message || 'Unable to finish workout session.');
            return null;
        } finally {
            setIsBusy(false);
        }
    };

    const value = {
        isAuthenticated: Boolean(authUser),
        authUser,
        authTokens,
        profile,
        isHydrating,
        needsOnboarding: Boolean(authUser) && !profile,
        goals,
        needsGoalsOnboarding: Boolean(authUser) && Boolean(profile) && !goals,
        themeMode,
        workoutPreferences,
        isBusy,
        latestError,
        currentSession,
        availableExercises,
        history,
        bodyMetrics,
        latestSummary,
        latestFeedback,
        latestFatiguePrediction,
        fatigueHistory,
        login,
        register,
        logout,
        loadCurrentUser,
        loadHistory,
        loadBodyMetrics,
        loadExercises,
        predictFatigue,
        loadFatigueHistory,
        saveProfile,
        saveGoals,
        saveWorkoutPreferences,
        updateProfileAvatar,
        addBodyMetricEntry,
        clearError,
        setLatestError,
        toggleThemeMode,
        startWorkout,
        updateMetrics,
        finishWorkout,
    };
    return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}
export function useAppState() {
    const context = useContext(AppStateContext);
    if (!context) {
        throw new Error('useAppState must be used inside AppStateProvider');
    }
    return context;
}
