import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

const LOADING_MS = 1000;
const featurePills = [
    { icon: 'scan-outline', label: 'Live form' },
    { icon: 'pulse-outline', label: 'Readiness' },
    { icon: 'trophy-outline', label: 'Progress' },
];

export function WelcomeScreen() {
    const router = useRouter();
    const { themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const [isLoading, setIsLoading] = useState(true);
    useEffect(() => {
        const timer = setTimeout(() => setIsLoading(false), LOADING_MS);
        return () => clearTimeout(timer);
    }, []);
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <View style={styles.container}>
        <View style={styles.hero}>
          <View style={[styles.logoMark, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <Ionicons name="body-outline" size={46} color={colors.brandAlt} />
          </View>
          <Text style={[styles.title, { color: colors.textPrimary }]}>CoachVision</Text>
          <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
            Your form, fatigue, and progress in one training cockpit.
          </Text>
          <View style={styles.pillRow}>
            {featurePills.map((item) => (
              <View key={item.label} style={[styles.pill, { backgroundColor: colors.surface, borderColor: colors.border }]}>
                <Ionicons name={item.icon} size={15} color={colors.brandAlt} />
                <Text style={[styles.pillText, { color: colors.textPrimary }]}>{item.label}</Text>
              </View>
            ))}
          </View>
        </View>

        <View style={[styles.previewPanel, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          <View style={styles.previewHeader}>
            <View>
              <Text style={[styles.previewLabel, { color: colors.textSecondary }]}>Today</Text>
              <Text style={[styles.previewTitle, { color: colors.textPrimary }]}>Squat readiness</Text>
            </View>
            <View style={[styles.scoreBubble, { backgroundColor: `${colors.success}22` }]}>
              <Text style={[styles.scoreText, { color: colors.success }]}>82</Text>
            </View>
          </View>
          <View style={[styles.previewTrack, { backgroundColor: colors.surfaceAlt }]}>
            <View style={[styles.previewFill, { backgroundColor: colors.brandAlt }]} />
          </View>
          <View style={styles.previewStats}>
            <PreviewStat icon="barbell-outline" label="Target" value="12 reps" colors={colors} />
            <PreviewStat icon="shield-checkmark-outline" label="Form" value="Ready" colors={colors} />
          </View>
        </View>

        {isLoading ? (<View style={styles.loadingBox}>
            <ActivityIndicator size="large" color={colors.brandAlt}/>
            <Text style={[styles.loadingText, { color: colors.textSecondary }]}>Preparing your space...</Text>
          </View>) : (<View style={styles.actions}>
            <PrimaryButton icon="log-in-outline" title="Sign in" onPress={() => router.push('/(auth)/login')}/>
            <PrimaryButton icon="person-add-outline" title="Create account" variant="secondary" onPress={() => router.push('/(auth)/register')}/>
          </View>)}
      </View>
    </SafeAreaView>);
}

function PreviewStat({ icon, label, value, colors }) {
    return (
      <View style={[styles.previewStat, { backgroundColor: colors.surfaceAlt }]}>
        <Ionicons name={icon} size={17} color={colors.brandAlt} />
        <View>
          <Text style={[styles.previewStatValue, { color: colors.textPrimary }]}>{value}</Text>
          <Text style={[styles.previewStatLabel, { color: colors.textSecondary }]}>{label}</Text>
        </View>
      </View>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { flex: 1, justifyContent: 'center', padding: 20, gap: 18 },
    hero: { alignItems: 'center', gap: 12 },
    logoMark: {
        width: 92,
        height: 92,
        borderRadius: 24,
        borderWidth: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    title: { fontSize: 38, lineHeight: 44, fontWeight: '900', textAlign: 'center' },
    subtitle: { fontSize: 16, lineHeight: 22, textAlign: 'center', maxWidth: 310 },
    pillRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, justifyContent: 'center' },
    pill: {
        minHeight: 34,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        borderRadius: 8,
        borderWidth: 1,
        paddingHorizontal: 10,
    },
    pillText: { fontSize: 12, fontWeight: '800' },
    previewPanel: {
        borderWidth: 1,
        borderRadius: 8,
        padding: 16,
        gap: 14,
    },
    previewHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
    previewLabel: { fontSize: 12, fontWeight: '800', textTransform: 'uppercase' },
    previewTitle: { fontSize: 18, fontWeight: '900', marginTop: 2 },
    scoreBubble: {
        width: 58,
        height: 58,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    scoreText: { fontSize: 24, fontWeight: '900' },
    previewTrack: { height: 10, borderRadius: 8, overflow: 'hidden' },
    previewFill: { width: '82%', height: '100%', borderRadius: 8 },
    previewStats: { flexDirection: 'row', gap: 10 },
    previewStat: {
        flex: 1,
        minHeight: 58,
        borderRadius: 8,
        padding: 10,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 10,
    },
    previewStatValue: { fontSize: 14, fontWeight: '900' },
    previewStatLabel: { fontSize: 11, fontWeight: '700', marginTop: 1 },
    loadingBox: { alignItems: 'center', gap: 12, minHeight: 112, justifyContent: 'center' },
    loadingText: { fontSize: 14, fontWeight: '700' },
    actions: { gap: 10 },
});
