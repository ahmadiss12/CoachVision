import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
export function FeedbackBanner({ message }) {
    const { themeMode } = useAppState();
    const palette = getThemeColors(themeMode);
    return (<View style={[styles.container, { borderColor: palette.border, backgroundColor: palette.surfaceAlt }]}>
      <Text style={[styles.label, { color: palette.textSecondary }]}>Live feedback</Text>
      <Text style={[styles.message, { color: palette.textPrimary }]}>{message}</Text>
    </View>);
}
const styles = StyleSheet.create({
    container: {
        borderRadius: 12,
        borderWidth: 1,
        padding: 12,
        gap: 4,
    },
    label: {
        fontSize: 12,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
    },
    message: {
        fontSize: 14,
        fontWeight: '500',
    },
});
