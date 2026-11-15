import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { planLabel } from '@/src/services/plan-access';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

/** Small pill naming the plan a feature belongs to, with a lock when it is not available. */
export function PlanTag({ plan, locked = false, style }) {
    const { themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    if (!plan) {
        return null;
    }
    const tint = locked ? colors.warning : colors.brandAlt;
    return (
      <View style={[styles.tag, { backgroundColor: colors.surface, borderColor: tint }, style]}>
        {locked ? <Ionicons name="lock-closed" size={10} color={tint} /> : null}
        <Text style={[styles.text, { color: tint }]}>{planLabel(plan).toUpperCase()}</Text>
      </View>
    );
}

const styles = StyleSheet.create({
    tag: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
        borderWidth: 1,
        borderRadius: 999,
        paddingHorizontal: 8,
        paddingVertical: 3,
        alignSelf: 'flex-start',
    },
    text: { fontSize: 10, fontWeight: '900', letterSpacing: 0.5 },
});
