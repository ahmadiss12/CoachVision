import React from 'react';
import { View } from 'react-native';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
export function Card({ style, ...props }) {
    const { themeMode } = useAppState();
    const palette = getThemeColors(themeMode);
    return (<View style={[
            {
                backgroundColor: palette.surface,
                borderRadius: 8,
                borderWidth: 1,
                borderColor: palette.border,
                padding: 16,
                shadowColor: '#000',
                shadowOpacity: themeMode === 'light' ? 0.08 : 0.18,
                shadowRadius: 10,
                shadowOffset: { width: 0, height: 6 },
                elevation: 2,
            },
            style,
        ]} {...props}/>);
}
