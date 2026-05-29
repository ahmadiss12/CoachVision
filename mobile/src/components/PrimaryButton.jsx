import React from 'react';
import { Pressable, StyleSheet, Text } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
export function PrimaryButton({
    title,
    onPress,
    variant = 'primary',
    disabled,
    icon,
    iconPosition = 'left',
    style,
    textStyle,
    ...props
}) {
    const { themeMode } = useAppState();
    const palette = getThemeColors(themeMode);
    const isDisabled = disabled === true || disabled === 'true';
    const isSecondary = variant === 'secondary';
    const isDanger = variant === 'danger';
    const isGhost = variant === 'ghost';
    const backgroundColor = isDanger
        ? palette.danger
        : isSecondary
            ? palette.surfaceAlt
            : isGhost
                ? 'transparent'
                : palette.brand;
    const foregroundColor = isSecondary || isGhost
        ? palette.textPrimary
        : '#FFFFFF';
    const iconNode = icon ? (
      <Ionicons name={icon} size={18} color={foregroundColor} />
    ) : null;
    return (<Pressable disabled={isDisabled} style={({ pressed }) => [
            styles.button,
            {
                backgroundColor,
                borderColor: isGhost ? 'transparent' : isSecondary ? palette.border : backgroundColor,
                opacity: isDisabled ? 0.5 : pressed ? 0.82 : 1,
            },
            style,
        ]} onPress={onPress} {...props}>
      {iconPosition === 'left' ? iconNode : null}
      <Text style={[styles.text, { color: foregroundColor }, textStyle]} numberOfLines={1}>
        {title}
      </Text>
      {iconPosition === 'right' ? iconNode : null}
    </Pressable>);
}
const styles = StyleSheet.create({
    button: {
        minHeight: 48,
        borderRadius: 8,
        borderWidth: 1,
        paddingVertical: 12,
        paddingHorizontal: 16,
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'row',
        gap: 8,
    },
    text: {
        fontSize: 16,
        fontWeight: '800',
    },
});
