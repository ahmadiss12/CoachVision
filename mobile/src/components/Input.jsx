import React from 'react';
import { StyleSheet, Text, TextInput, View } from 'react-native';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
export function Input({ label, value, placeholder, secureTextEntry, keyboardType, autoCapitalize = 'none', editable = true, onChangeText, }) {
    const { themeMode } = useAppState();
    const palette = getThemeColors(themeMode);
    return (<View style={styles.container}>
      <Text style={[styles.label, { color: palette.textSecondary }]}>{label}</Text>
      <TextInput value={value} placeholder={placeholder} placeholderTextColor={palette.textSecondary} secureTextEntry={secureTextEntry} keyboardType={keyboardType} autoCapitalize={autoCapitalize} editable={editable} style={[
            styles.input,
            {
                borderColor: palette.border,
                backgroundColor: palette.surface,
                color: palette.textPrimary,
            },
        ]} onChangeText={onChangeText}/>
    </View>);
}
const styles = StyleSheet.create({
    container: { gap: 8 },
    label: { fontSize: 14 },
    input: {
        borderWidth: 1,
        borderRadius: 12,
        paddingHorizontal: 14,
        paddingVertical: 12,
        fontSize: 16,
    },
});
