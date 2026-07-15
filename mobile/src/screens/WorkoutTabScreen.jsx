import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card } from '@/src/components/Card';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

const flows = [
    {
        icon: 'play-circle-outline',
        title: 'Guided live session',
        text: 'Camera form tracking, voice cues, and session summary.',
        route: '/(app)/workout-setup',
        tone: 'brandAlt',
    },
    {
        icon: 'time-outline',
        title: 'Past sessions',
        text: 'Review scores, reps, duration, and workout notes.',
        route: '/(app)/history',
        tone: 'info',
    },
    {
        icon: 'analytics-outline',
        title: 'Body progress',
        text: 'Log weight and body composition check-ins.',
        route: '/(app)/body-progress',
        tone: 'success',
    },
];

export function WorkoutTabScreen() {
    const router = useRouter();
    const { themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <View style={styles.container}>
        <ScreenHeader
          icon="barbell-outline"
          title="Workout"
          subtitle="Choose the training flow you need right now."
        />

        <Card style={styles.quickCard}>
          <View style={styles.quickHeader}>
            <View style={[styles.quickIcon, { backgroundColor: `${colors.brandAlt}22` }]}>
              <Ionicons name="flash-outline" size={24} color={colors.brandAlt} />
            </View>
            <View style={styles.quickCopy}>
              <Text style={[styles.quickTitle, { color: colors.textPrimary }]}>Quick start</Text>
              <Text style={[styles.quickText, { color: colors.textSecondary }]}>Begin with exercise setup and readiness target.</Text>
            </View>
          </View>
          <PrimaryButton icon="play-outline" title="Start new workout" onPress={() => router.push('/(app)/workout-setup')}/>
        </Card>

        <View style={styles.flowList}>
          {flows.map((item) => (
            <FlowRow key={item.title} item={item} colors={colors} onPress={() => router.push(item.route)} />
          ))}
        </View>
      </View>
    </SafeAreaView>);
}

function FlowRow({ item, colors, onPress }) {
    const toneColor = colors[item.tone] ?? colors.brandAlt;
    return (
      <Pressable
        onPress={onPress}
        style={({ pressed }) => [
          styles.flowRow,
          { backgroundColor: colors.surface, borderColor: colors.border, opacity: pressed ? 0.86 : 1 },
        ]}
      >
        <View style={[styles.flowIcon, { backgroundColor: `${toneColor}22` }]}>
          <Ionicons name={item.icon} size={21} color={toneColor} />
        </View>
        <View style={styles.flowCopy}>
          <Text style={[styles.flowTitle, { color: colors.textPrimary }]}>{item.title}</Text>
          <Text style={[styles.flowText, { color: colors.textSecondary }]}>{item.text}</Text>
        </View>
        <Ionicons name="chevron-forward" size={18} color={colors.textSecondary} />
      </Pressable>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { flex: 1, padding: 20, gap: 14 },
    quickCard: { gap: 14 },
    quickHeader: { flexDirection: 'row', alignItems: 'center', gap: 12 },
    quickIcon: {
        width: 48,
        height: 48,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    quickCopy: { flex: 1, minWidth: 0 },
    quickTitle: { fontSize: 18, fontWeight: '900' },
    quickText: { marginTop: 2, lineHeight: 20 },
    flowList: { gap: 10 },
    flowRow: {
        minHeight: 86,
        borderWidth: 1,
        borderRadius: 8,
        padding: 14,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
    },
    flowIcon: {
        width: 44,
        height: 44,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    flowCopy: { flex: 1, minWidth: 0 },
    flowTitle: { fontSize: 15, fontWeight: '900' },
    flowText: { fontSize: 13, lineHeight: 18, marginTop: 2 },
});
