import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { planLabel } from '@/src/services/plan-access';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

/** Says what a plan would add in this spot and links to the plans screen. */
export function LockedNotice({ plan, title, message, style }) {
    const router = useRouter();
    const { themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const heading = title ?? `Included with ${planLabel(plan)}`;
    return (
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={`${heading}. See plans.`}
        onPress={() => router.push('/(app)/plans')}
        style={({ pressed }) => [
            styles.box,
            { backgroundColor: colors.surfaceAlt, borderColor: colors.border, opacity: pressed ? 0.85 : 1 },
            style,
        ]}
      >
        <View style={[styles.icon, { backgroundColor: `${colors.warning}22` }]}>
          <Ionicons name="lock-closed" size={16} color={colors.warning} />
        </View>
        <View style={styles.copy}>
          <Text style={[styles.title, { color: colors.textPrimary }]}>{heading}</Text>
          {message ? <Text style={[styles.message, { color: colors.textSecondary }]}>{message}</Text> : null}
        </View>
        <View style={styles.link}>
          <Text style={[styles.linkText, { color: colors.brandAlt }]}>See plans</Text>
          <Ionicons name="chevron-forward" size={14} color={colors.brandAlt} />
        </View>
      </Pressable>
    );
}

const styles = StyleSheet.create({
    box: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 10,
        borderWidth: 1,
        borderRadius: 8,
        padding: 12,
    },
    icon: { width: 32, height: 32, borderRadius: 16, alignItems: 'center', justifyContent: 'center' },
    copy: { flex: 1, gap: 2 },
    title: { fontSize: 14, fontWeight: '800' },
    message: { fontSize: 12, lineHeight: 17 },
    link: { flexDirection: 'row', alignItems: 'center', gap: 2 },
    linkText: { fontSize: 12, fontWeight: '800' },
});
