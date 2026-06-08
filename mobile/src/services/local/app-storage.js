import * as SecureStore from 'expo-secure-store';

const AUTH_TOKENS_KEY = 'coachvision.auth.tokens.v1';
const THEME_MODE_KEY = 'coachvision.themeMode.v1';
const GOALS_KEY_PREFIX = 'coachvision.goals.v1.';
const WORKOUT_PREFS_KEY_PREFIX = 'coachvision.workoutPrefs.v1.';

const memoryFallback = new Map();

function safeUserKey(user) {
  const raw = String(user?.id ?? user?.email ?? 'unknown');
  return raw.replace(/[^A-Za-z0-9._-]/g, '_');
}

async function getItem(key) {
  try {
    return await SecureStore.getItemAsync(key);
  } catch {
    return memoryFallback.get(key) ?? null;
  }
}

async function setItem(key, value) {
  memoryFallback.set(key, value);
  try {
    await SecureStore.setItemAsync(key, value);
  } catch {
    // In unsupported environments, the in-memory fallback keeps this session usable.
  }
}

async function deleteItem(key) {
  memoryFallback.delete(key);
  try {
    await SecureStore.deleteItemAsync(key);
  } catch {
    // Ignore storage cleanup failures; the auth state is cleared in memory too.
  }
}

async function readJson(key) {
  const raw = await getItem(key);
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function writeJson(key, value) {
  return setItem(key, JSON.stringify(value));
}

export function userStorageKey(user) {
  return safeUserKey(user);
}

export async function loadStoredAuthTokens() {
  return readJson(AUTH_TOKENS_KEY);
}

export async function saveStoredAuthTokens(tokens) {
  if (!tokens) {
    await clearStoredAuthTokens();
    return;
  }
  await writeJson(AUTH_TOKENS_KEY, tokens);
}

export async function clearStoredAuthTokens() {
  await deleteItem(AUTH_TOKENS_KEY);
}

export async function loadStoredThemeMode() {
  const value = await getItem(THEME_MODE_KEY);
  return value === 'light' || value === 'dark' ? value : null;
}

export async function saveStoredThemeMode(themeMode) {
  if (themeMode !== 'light' && themeMode !== 'dark') {
    return;
  }
  await setItem(THEME_MODE_KEY, themeMode);
}

export async function loadStoredGoals(userKey) {
  if (!userKey) {
    return null;
  }
  return readJson(`${GOALS_KEY_PREFIX}${userKey}`);
}

export async function saveStoredGoals(userKey, goals) {
  if (!userKey || !goals) {
    return;
  }
  await writeJson(`${GOALS_KEY_PREFIX}${userKey}`, goals);
}

export async function loadStoredWorkoutPreferences(userKey) {
  if (!userKey) {
    return null;
  }
  return readJson(`${WORKOUT_PREFS_KEY_PREFIX}${userKey}`);
}

export async function saveStoredWorkoutPreferences(userKey, preferences) {
  if (!userKey || !preferences) {
    return;
  }
  await writeJson(`${WORKOUT_PREFS_KEY_PREFIX}${userKey}`, preferences);
}
