import React, { createContext, useContext, useState } from 'react';
import {
  loginUser,
  logoutUser,
  refreshAuthToken,
  registerUser,
} from '../services/api/auth';
import { clearApiAuthTokens, setApiAuthTokens } from '../services/api/client';
import { listExercises } from '../services/api/exercises';
import {
  createSession,
  endSession,
  getSessionFeedback,
  startSession,
} from '../services/api/sessions';
import { getCurrentUser, updateCurrentUser } from '../services/api/users';
const defaultMetrics = {
    count: 0,
    state: 'idle',
    angle: 180,
    feedback: 'Start your set when ready.',
    progress: 0,
    formName: 'Correct',
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
export function AppStateProvider({ children }) {
    const [authUser, setAuthUser] = useState(null);
    const [authTokens, setAuthTokens] = useState(null);
    const [profile, setProfile] = useState(null);
    const [goals, setGoals] = useState(null);
    const [themeMode, setThemeMode] = useState('dark');
    const [isBusy, setIsBusy] = useState(false);
    const [latestError, setLatestError] = useState(null);
    const [currentSession, setCurrentSession] = useState(null);
    const [latestSummary, setLatestSummary] = useState(null);
    const [latestFeedback, setLatestFeedback] = useState(null);
    const [availableExercises, setAvailableExercises] = useState([]);
    const [history, setHistory] = useState([]);
    const [bodyMetrics, setBodyMetrics] = useState([]);

    const setTokensEverywhere = (tokens) => {
        setAuthTokens(tokens);
        setApiAuthTokens(tokens);
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
            clearApiAuthTokens();
            setAuthUser(null);
            return false;
        }
    };

    const loadCurrentUser = async () => {
        try {
            const user = await getCurrentUser();
            setAuthUser(user);
            setLatestError(null);
            return user;
        } catch (error) {
            if (error?.status === 401) {
                const refreshed = await ensureFreshAccessToken();
                if (refreshed) {
                    const user = await getCurrentUser();
                    setAuthUser(user);
                    setLatestError(null);
                    return user;
                }
            }
            throw error;
        }
    };

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
            const user = await loadCurrentUser();
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
            const user = await loadCurrentUser();
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
        clearApiAuthTokens();
        setAuthUser(null);
        setProfile(null);
        setGoals(null);
        setBodyMetrics([]);
        setCurrentSession(null);
        setLatestFeedback(null);
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
            return [
                {
                    id: `${Date.now()}`,
                    date: new Date().toISOString().slice(0, 10),
                    weightKg: safeWeightKg,
                    bodyFatPercent: safeBodyFatPercent,
                },
            ];
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
        return next;
    };
    const addBodyMetricEntry = (payload) => {
        const safeWeight = payload.weightKg > 0 ? payload.weightKg : profile?.weightKg ?? 70;
        const safeBodyFat = payload.bodyFatPercent >= 0 ? payload.bodyFatPercent : profile?.bodyFatPercent ?? 20;
        const date = payload.date || new Date().toISOString().slice(0, 10);
        setBodyMetrics((prev) => [
            {
                id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
                date,
                weightKg: safeWeight,
                bodyFatPercent: safeBodyFat,
            },
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
    };
    const clearError = () => setLatestError(null);
    const toggleThemeMode = () => {
        setThemeMode((prev) => (prev === 'dark' ? 'light' : 'dark'));
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
            const created = await createSession({
                exerciseId: config.exerciseName,
                difficulty: config.difficulty || 'beginner',
                targetSets: config.targetSets || 1,
                targetReps: config.targetReps || 1,
            });
            const started = await startSession(created.id);
            const nextSession = {
                config,
                sessionId: started.id,
                startedAt: started.startedAt || new Date().toISOString(),
                latestMetrics: defaultMetrics,
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
            const ended = await endSession(currentSession.sessionId);
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
            const durationSec = Math.max(1, Math.round((endedMs - startedMs) / 1000));
            const reps = endedSummary?.totalReps ?? currentSession.latestMetrics.count ?? 0;
            const score = feedback?.overallRating ?? Math.min(100, Math.round(70 + (currentSession.latestMetrics.progress || 0) * 30));

            const summary = {
                id: ended.id,
                sessionId: ended.id,
                startedAt,
                endedAt,
                durationSec,
                exerciseName: currentSession.config.exerciseName,
                difficulty: currentSession.config.difficulty,
                reps,
                score,
                notes: feedback?.summaryText || currentSession.latestMetrics.feedback,
                feedback,
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
        needsOnboarding: Boolean(authUser) && !profile,
        goals,
        needsGoalsOnboarding: Boolean(authUser) && Boolean(profile) && !goals,
        themeMode,
        isBusy,
        latestError,
        currentSession,
        availableExercises,
        history,
        bodyMetrics,
        latestSummary,
        latestFeedback,
        login,
        register,
        logout,
        loadCurrentUser,
        loadExercises,
        saveProfile,
        saveGoals,
        updateProfileAvatar,
        addBodyMetricEntry,
        clearError,
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
