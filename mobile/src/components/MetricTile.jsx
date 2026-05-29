import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { IconBadge } from '@/src/components/IconBadge';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

export function MetricTile({ icon, label, value, tone = 'brand', style }) {
    const { themeMode } = useAppState();
    const palette = getThemeColors(themeMode);
    const toneColor = tone === 'success'
        ? palette.success
        : tone === 'warning'
            ? palette.warning
            : tone === 'danger'
                ? palette.danger
                : tone === 'info'
                    ? palette.info
                    : palette.brandAlt;
    return (
      <View style={[styles.tile, { backgroundColor: palette.surface, borderColor: palette.border }, style]}>
        <IconBadge name={icon} color={toneColor} style={styles.icon} size={18} />
        <Text style={[styles.value, { color: palette.textPrimary }]} numberOfLines={1}>{value}</Text>
        <Text style={[styles.label, { color: palette.textSecondary }]} numberOfLines={2}>{label}</Text>
      </View>
    );
}

const styles = StyleSheet.create({
    tile: {
        flex: 1,
        minHeight: 112,
        borderWidth: 1,
        borderRadius: 8,
        padding: 12,
        justifyContent: 'space-between',
    },
    icon: {
        width: 36,
        height: 36,
    },
    value: {
        fontSize: 23,
        lineHeight: 28,
        fontWeight: '900',
    },
    label: {
        fontSize: 12,
        lineHeight: 16,
        fontWeight: '700',
    },
});
