import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
export function FeedbackBanner({ message, title = 'Live feedback', tone = 'info' }) {
    const { themeMode } = useAppState();
    const palette = getThemeColors(themeMode);
    const toneColor = tone === 'success'
        ? palette.success
        : tone === 'danger'
            ? palette.danger
            : tone === 'warning'
                ? palette.warning
                : palette.brandAlt;
    return (<View style={[styles.container, { borderColor: palette.border, backgroundColor: palette.surfaceAlt }]}>
      <View style={[styles.iconWrap, { backgroundColor: `${toneColor}22` }]}>
        <Ionicons name="sparkles-outline" size={18} color={toneColor} />
      </View>
      <View style={styles.copy}>
        <Text style={[styles.label, { color: palette.textSecondary }]}>{title}</Text>
        <Text style={[styles.message, { color: palette.textPrimary }]}>{message}</Text>
      </View>
    </View>);
}
const styles = StyleSheet.create({
    container: {
        borderRadius: 8,
        borderWidth: 1,
        padding: 12,
        gap: 10,
        flexDirection: 'row',
        alignItems: 'center',
    },
    iconWrap: {
        width: 38,
        height: 38,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    copy: {
        flex: 1,
        gap: 3,
    },
    label: {
        fontSize: 12,
        textTransform: 'uppercase',
        fontWeight: '800',
    },
    message: {
        fontSize: 14,
        fontWeight: '700',
        lineHeight: 19,
    },
});
