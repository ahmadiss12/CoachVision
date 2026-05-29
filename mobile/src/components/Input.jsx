import React, { useState } from 'react';
import { StyleSheet, Text, TextInput, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
export function Input({
    label,
    value,
    placeholder,
    secureTextEntry,
    keyboardType,
    autoCapitalize = 'none',
    editable = true,
    onChangeText,
    icon,
    ...props
}) {
    const { themeMode } = useAppState();
    const palette = getThemeColors(themeMode);
    const [isFocused, setIsFocused] = useState(false);
    return (<View style={styles.container}>
      <Text style={[styles.label, { color: palette.textSecondary }]}>{label}</Text>
      <View style={[
            styles.inputShell,
            {
                borderColor: isFocused ? palette.brandAlt : palette.border,
                backgroundColor: palette.surface,
            },
        ]}>
        {icon ? <Ionicons name={icon} size={18} color={isFocused ? palette.brandAlt : palette.textSecondary} /> : null}
        <TextInput
          value={value}
          placeholder={placeholder}
          placeholderTextColor={palette.textSecondary}
          secureTextEntry={secureTextEntry}
          keyboardType={keyboardType}
          autoCapitalize={autoCapitalize}
          editable={editable}
          style={[styles.input, { color: palette.textPrimary }]}
          onChangeText={onChangeText}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          {...props}
        />
      </View>
    </View>);
}
const styles = StyleSheet.create({
    container: { gap: 8 },
    label: { fontSize: 13, fontWeight: '700' },
    inputShell: {
        minHeight: 48,
        borderWidth: 1,
        borderRadius: 8,
        paddingHorizontal: 14,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 10,
    },
    input: {
        flex: 1,
        paddingVertical: 12,
        fontSize: 16,
        minWidth: 0,
    },
});
