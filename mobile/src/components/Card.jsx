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
                borderRadius: 16,
                borderWidth: 1,
                borderColor: palette.border,
                padding: 16,
            },
            style,
        ]} {...props}/>);
}
