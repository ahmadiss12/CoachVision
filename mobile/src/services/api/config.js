import Constants from 'expo-constants';
import { Platform } from 'react-native';

const trimTrailingSlash = (value) => value.replace(/\/+$/, '');

const withDefault = (value, fallback) => {
  if (typeof value !== 'string') {
    return fallback;
  }
  const normalized = value.trim();
  return normalized.length > 0 ? normalized : fallback;
};

const getExpoDevHost = () => {
  const hostUri = Constants.expoConfig?.hostUri || Constants.manifest2?.extra?.expoClient?.hostUri;
  if (typeof hostUri !== 'string') {
    return '';
  }
  return hostUri.split(':')[0] || '';
};

const normalizeOriginForPlatform = (origin) => {
  if (Platform.OS === 'web') {
    return origin;
  }
  if (!/^(https?|wss?):\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(origin)) {
    return origin;
  }

  const devHost = getExpoDevHost();
  if (!devHost) {
    return origin;
  }
  const url = new URL(origin);
  url.hostname = devHost;
  return url.toString();
};

const API_ORIGIN = trimTrailingSlash(
  normalizeOriginForPlatform(withDefault(process.env.EXPO_PUBLIC_API_ORIGIN, 'http://127.0.0.1:8001'))
);
const API_PREFIX = withDefault(process.env.EXPO_PUBLIC_API_PREFIX, '/v1');

const configuredWsOrigin = normalizeOriginForPlatform(withDefault(process.env.EXPO_PUBLIC_WS_ORIGIN, ''));
const derivedWsOrigin = API_ORIGIN.replace(/^http/i, 'ws');
const WS_ORIGIN = trimTrailingSlash(configuredWsOrigin || derivedWsOrigin);

// Trainer-facing web dashboard. Deliberately not normalized for the dev host:
// this is a public site, not the local backend, so it stays as configured.
const WEB_DASHBOARD_URL = trimTrailingSlash(
  withDefault(
    process.env.EXPO_PUBLIC_WEB_DASHBOARD_URL,
    'https://web-ahmadiss12s-projects.vercel.app'
  )
);

export const apiConfig = {
  apiOrigin: API_ORIGIN,
  apiPrefix: API_PREFIX.startsWith('/') ? API_PREFIX : `/${API_PREFIX}`,
  wsOrigin: WS_ORIGIN,
  webDashboardUrl: WEB_DASHBOARD_URL,
};

export function buildApiUrl(path) {
  const safePath = path.startsWith('/') ? path : `/${path}`;
  return `${apiConfig.apiOrigin}${apiConfig.apiPrefix}${safePath}`;
}

export function buildWsUrl(path, query = {}) {
  const safePath = path.startsWith('/') ? path : `/${path}`;
  const url = new URL(`${apiConfig.wsOrigin}${apiConfig.apiPrefix}${safePath}`);
  Object.entries(query).forEach(([key, value]) => {
    if (value === undefined || value === null) {
      return;
    }
    url.searchParams.set(key, String(value));
  });
  return url.toString();
}
