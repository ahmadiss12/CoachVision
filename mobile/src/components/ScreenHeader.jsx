import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { IconBadge } from '@/src/components/IconBadge';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

export function ScreenHeader({ title, subtitle, icon, eyebrow, right }) {
    const { themeMode } = useAppState();
    const palette = getThemeColors(themeMode);
    return (
      <View style={styles.header}>
        <View style={styles.titleRow}>
          {icon ? <IconBadge name={icon} /> : null}
          <View style={styles.copy}>
            {eyebrow ? <Text style={[styles.eyebrow, { color: palette.brandAlt }]}>{eyebrow}</Text> : null}
            <Text style={[styles.title, { color: palette.textPrimary }]}>{title}</Text>
          </View>
          {right}
        </View>
        {subtitle ? <Text style={[styles.subtitle, { color: palette.textSecondary }]}>{subtitle}</Text> : null}
      </View>
    );
}

const styles = StyleSheet.create({
    header: {
        gap: 8,
        marginTop: 8,
    },
    titleRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
    },
    copy: {
        flex: 1,
        minWidth: 0,
    },
    eyebrow: {
        fontSize: 12,
        fontWeight: '900',
        textTransform: 'uppercase',
        marginBottom: 2,
    },
    title: {
        fontSize: 29,
        lineHeight: 34,
        fontWeight: '900',
    },
    subtitle: {
        fontSize: 15,
        lineHeight: 21,
    },
});
