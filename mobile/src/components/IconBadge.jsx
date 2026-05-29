import React from 'react';
import { StyleSheet, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

export function IconBadge({ name, color, backgroundColor, size = 20, style }) {
    const { themeMode } = useAppState();
    const palette = getThemeColors(themeMode);
    const iconColor = color ?? palette.brandAlt;
    return (
      <View style={[styles.badge, { backgroundColor: backgroundColor ?? `${iconColor}22` }, style]}>
        <Ionicons name={name} size={size} color={iconColor} />
      </View>
    );
}

const styles = StyleSheet.create({
    badge: {
        width: 42,
        height: 42,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
});
