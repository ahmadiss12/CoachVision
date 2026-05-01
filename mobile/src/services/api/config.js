const trimTrailingSlash = (value) => value.replace(/\/+$/, '');

const withDefault = (value, fallback) => {
  if (typeof value !== 'string') {
    return fallback;
  }
  const normalized = value.trim();
  return normalized.length > 0 ? normalized : fallback;
};

const API_ORIGIN = trimTrailingSlash(
  withDefault(process.env.EXPO_PUBLIC_API_ORIGIN, 'http://127.0.0.1:8000')
);
const API_PREFIX = withDefault(process.env.EXPO_PUBLIC_API_PREFIX, '/v1');

const configuredWsOrigin = withDefault(process.env.EXPO_PUBLIC_WS_ORIGIN, '');
const derivedWsOrigin = API_ORIGIN.replace(/^http/i, 'ws');
const WS_ORIGIN = trimTrailingSlash(configuredWsOrigin || derivedWsOrigin);

export const apiConfig = {
  apiOrigin: API_ORIGIN,
  apiPrefix: API_PREFIX.startsWith('/') ? API_PREFIX : `/${API_PREFIX}`,
  wsOrigin: WS_ORIGIN,
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
