const darkColors = {
    background: '#0B1220',
    surface: '#121A2B',
    surfaceAlt: '#1A2438',
    textPrimary: '#F1F5F9',
    textSecondary: '#94A3B8',
    brand: '#4F46E5',
    brandAlt: '#6366F1',
    success: '#22C55E',
    warning: '#F59E0B',
    danger: '#EF4444',
    border: '#273246',
};
const lightColors = {
    background: '#F3F7FF',
    surface: '#FFFFFF',
    surfaceAlt: '#E8EEFC',
    textPrimary: '#0F172A',
    textSecondary: '#475569',
    brand: '#4F46E5',
    brandAlt: '#6366F1',
    success: '#16A34A',
    warning: '#D97706',
    danger: '#DC2626',
    border: '#CBD5E1',
};
export const colors = darkColors;
export function getThemeColors(mode) {
    return mode === 'light' ? lightColors : darkColors;
}
