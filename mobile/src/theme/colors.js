const darkColors = {
    background: '#101312',
    surface: '#171B19',
    surfaceAlt: '#222823',
    textPrimary: '#F6F7F3',
    textSecondary: '#AEB7B0',
    brand: '#0F766E',
    brandAlt: '#2DD4BF',
    success: '#22C55E',
    warning: '#F59E0B',
    danger: '#EF4444',
    info: '#38BDF8',
    border: '#34413B',
};
const lightColors = {
    background: '#F7F8F3',
    surface: '#FFFFFF',
    surfaceAlt: '#ECF2EA',
    textPrimary: '#18201C',
    textSecondary: '#58635C',
    brand: '#0F766E',
    brandAlt: '#0D9488',
    success: '#16A34A',
    warning: '#D97706',
    danger: '#DC2626',
    info: '#0284C7',
    border: '#CDD8D0',
};
export const colors = darkColors;
export function getThemeColors(mode) {
    return mode === 'light' ? lightColors : darkColors;
}
