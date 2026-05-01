import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
export function ProgressBar({ value }) {
    const { themeMode } = useAppState();
    const palette = getThemeColors(themeMode);
    const normalized = Math.max(0, Math.min(1, value));
    return (<View style={styles.wrapper}>
      <View style={[styles.track, { backgroundColor: palette.surfaceAlt }]}>
        <View style={[styles.fill, { width: `${normalized * 100}%`, backgroundColor: palette.success }]}/>
      </View>
      <Text style={[styles.label, { color: palette.textSecondary }]}>{Math.round(normalized * 100)}%</Text>
    </View>);
}
const styles = StyleSheet.create({
    wrapper: { gap: 8 },
    track: {
        borderRadius: 999,
        height: 10,
        overflow: 'hidden',
    },
    fill: {
        height: '100%',
        borderRadius: 999,
    },
    label: {
        fontSize: 12,
        textAlign: 'right',
    },
});
