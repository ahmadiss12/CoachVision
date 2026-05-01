import React from 'react';
import { Pressable, StyleSheet, Text } from 'react-native';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
export function PrimaryButton({ title, onPress, variant = 'primary', disabled }) {
    const { themeMode } = useAppState();
    const palette = getThemeColors(themeMode);
    const isDisabled = disabled === true || disabled === 'true';
    const backgroundColor = variant === 'primary' ? palette.brand : variant === 'danger' ? palette.danger : palette.surfaceAlt;
    return (<Pressable disabled={isDisabled} style={({ pressed }) => [
            styles.button,
            { backgroundColor, opacity: isDisabled ? 0.5 : pressed ? 0.85 : 1 },
        ]} onPress={onPress}>
      <Text style={[styles.text, { color: themeMode === 'light' ? '#FFFFFF' : palette.textPrimary }]}>{title}</Text>
    </Pressable>);
}
const styles = StyleSheet.create({
    button: {
        borderRadius: 12,
        paddingVertical: 12,
        paddingHorizontal: 16,
        alignItems: 'center',
        justifyContent: 'center',
    },
    text: {
        fontSize: 16,
        fontWeight: '600',
    },
});
