import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card } from '@/src/components/Card';
import { IconBadge } from '@/src/components/IconBadge';
import { MetricTile } from '@/src/components/MetricTile';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { TodayPlanCard } from '@/src/components/TodayPlanCard';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

function buildSuggestedPlan(input) {
    const { targetWeightLeft, currentBodyFat, targetBodyFat } = input;
    const bodyFatGap = currentBodyFat !== null && targetBodyFat !== null ? currentBodyFat - targetBodyFat : 0;
    const needsFatLoss = (targetWeightLeft !== null && targetWeightLeft > 1) || bodyFatGap > 1.5;
    const needsGain = targetWeightLeft !== null && targetWeightLeft < -1;
    if (needsFatLoss) {
        return {
            focus: 'Fat loss + endurance',
            weeklySessions: 5,
            icon: 'flame-outline',
            plan: [
                { name: 'Circuit Squat', sets: 4, reps: '15', intensity: 'High', icon: 'body-outline' },
                { name: 'Push-up', sets: 4, reps: '12-15', intensity: 'Moderate', icon: 'hand-left-outline' },
                { name: 'High Knees', sets: 5, reps: '40 sec', intensity: 'High', icon: 'trending-up-outline' },
            ],
        };
    }
    if (needsGain) {
        return {
            focus: 'Strength + muscle gain',
            weeklySessions: 4,
            icon: 'barbell-outline',
            plan: [
                { name: 'Squat', sets: 5, reps: '8-10', intensity: 'High', icon: 'body-outline' },
                { name: 'Lunge', sets: 4, reps: '10 each leg', intensity: 'Moderate', icon: 'walk-outline' },
                { name: 'Plank', sets: 4, reps: '45 sec', intensity: 'Moderate', icon: 'male-outline' },
            ],
        };
    }
    return {
        focus: 'Recomposition + consistency',
        weeklySessions: 4,
        icon: 'repeat-outline',
        plan: [
            { name: 'Squat', sets: 4, reps: '12', intensity: 'Moderate', icon: 'body-outline' },
            { name: 'Push-up', sets: 4, reps: '10-12', intensity: 'Moderate', icon: 'hand-left-outline' },
            { name: 'Jumping Jack', sets: 4, reps: '30 sec', intensity: 'Low', icon: 'flash-outline' },
        ],
    };
}

export function HomeScreen() {
    const router = useRouter();
    const { authUser, history, latestSummary, profile, goals, bodyMetrics, themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const userLabel = authUser?.email ? authUser.email.split('@')[0] : 'athlete';
    const recentScore = latestSummary?.score ?? 0;
    const recentReps = latestSummary?.reps ?? 0;
    const now = new Date();
    const weekAgoMs = now.getTime() - 7 * 24 * 60 * 60 * 1000;
    const recentSessions = history.filter((session) => new Date(session.endedAt).getTime() >= weekAgoMs).length;
    const weeklyTarget = goals?.weeklyWorkoutTarget ?? 4;
    const weeklyProgress = Math.min(1, recentSessions / Math.max(1, weeklyTarget));
    const targetWeightLeft = profile && goals ? Number((profile.weightKg - goals.targetWeightKg).toFixed(1)) : null;
    const latestBodyEntry = bodyMetrics[0];
    const shouldRemindBodyUpdate = !latestBodyEntry
        ? true
        : now.getTime() - new Date(latestBodyEntry.date).getTime() > 7 * 24 * 60 * 60 * 1000;
    const shouldRemindWorkout = history.length === 0;
    const suggestion = buildSuggestedPlan({
        targetWeightLeft,
        currentBodyFat: profile?.bodyFatPercent ?? null,
        targetBodyFat: goals?.targetBodyFatPercent ?? null,
    });
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>
        <ScreenHeader
          icon="home-outline"
          eyebrow={`Hi ${userLabel}`}
          title="Training dashboard"
          subtitle="A quick view of readiness, progress, and your next move."
        />

        <TodayPlanCard colors={colors} />

        <Card style={[styles.heroCard, { backgroundColor: colors.surface }]}>
          <View style={styles.heroTop}>
            <IconBadge name={suggestion.icon} color={colors.warning} />
            <View style={styles.heroCopy}>
              <Text style={[styles.heroLabel, { color: colors.textSecondary }]}>Next focus</Text>
              <Text style={[styles.heroTitle, { color: colors.textPrimary }]}>{suggestion.focus}</Text>
            </View>
            <View style={[styles.weekBadge, { backgroundColor: colors.surfaceAlt }]}>
              <Text style={[styles.weekValue, { color: colors.brandAlt }]}>{recentSessions}/{weeklyTarget}</Text>
              <Text style={[styles.weekLabel, { color: colors.textSecondary }]}>week</Text>
            </View>
          </View>
          <View style={[styles.progressTrack, { backgroundColor: colors.surfaceAlt }]}>
            <View style={[styles.progressFill, { width: `${weeklyProgress * 100}%`, backgroundColor: colors.brandAlt }]} />
          </View>
          <View style={styles.heroActions}>
            <PrimaryButton icon="play-circle-outline" title="Start workout" onPress={() => router.push('/(app)/workout-setup')} style={styles.heroButton}/>
            <PrimaryButton icon="time-outline" title="History" variant="secondary" onPress={() => router.push('/(app)/history')} style={styles.heroButton}/>
          </View>
        </Card>

        <View style={styles.metricGrid}>
          <View style={styles.metricRow}>
            <MetricTile icon="calendar-number-outline" label="Total sessions" value={history.length} tone="brand" />
            <MetricTile icon="pulse-outline" label="Latest score" value={recentScore} tone="success" />
          </View>
          <View style={styles.metricRow}>
            <MetricTile icon="repeat-outline" label="Latest output" value={recentReps} tone="warning" />
            <MetricTile icon="flag-outline" label="Weekly goal" value={`${recentSessions}/${weeklyTarget}`} tone="info" />
          </View>
        </View>

        <Card style={styles.sectionCard}>
          <SectionTitle icon="flag-outline" title="Goals snapshot" colors={colors} />
          <InfoLine
            icon="scale-outline"
            label="Target weight left"
            value={targetWeightLeft === null ? '--' : `${targetWeightLeft > 0 ? '+' : ''}${targetWeightLeft} kg`}
            tone={targetWeightLeft !== null && targetWeightLeft > 0 ? colors.warning : colors.success}
            colors={colors}
          />
          <InfoLine
            icon="calendar-outline"
            label="Weekly progress"
            value={`${recentSessions}/${weeklyTarget} sessions`}
            tone={recentSessions >= weeklyTarget ? colors.success : colors.brandAlt}
            colors={colors}
          />
          <PrimaryButton icon="create-outline" title="Update goals" variant="secondary" onPress={() => router.push('/(app)/goals')}/>
        </Card>

        <Card style={styles.sectionCard}>
          <SectionTitle icon="notifications-outline" title="Reminders" colors={colors} />
          <ReminderRow
            icon={shouldRemindBodyUpdate ? 'scale-outline' : 'checkmark-circle-outline'}
            text={shouldRemindBodyUpdate ? 'Update your weight and body fat check-in.' : 'Body data is up to date this week.'}
            color={shouldRemindBodyUpdate ? colors.warning : colors.success}
            colors={colors}
          />
          <ReminderRow
            icon={shouldRemindWorkout ? 'barbell-outline' : 'flame-outline'}
            text={shouldRemindWorkout ? 'Start your first logged workout.' : 'Keep your training rhythm alive.'}
            color={shouldRemindWorkout ? colors.warning : colors.success}
            colors={colors}
          />
          <PrimaryButton icon="analytics-outline" title="Update body data" variant="secondary" onPress={() => router.push('/(app)/body-progress')}/>
        </Card>

        <Card style={styles.sectionCard}>
          <SectionTitle icon="sparkles-outline" title="Suggested training" colors={colors} />
          <Text style={[styles.sectionNote, { color: colors.textSecondary }]}>
            {suggestion.weeklySessions} sessions this week | Goal based
          </Text>
          {suggestion.plan.map((item) => (
            <TrainingRow key={item.name} item={item} colors={colors} />
          ))}
          <PrimaryButton icon="arrow-forward-circle-outline" iconPosition="right" title="Start now" onPress={() => router.push('/(app)/workout-setup')}/>
        </Card>
      </ScrollView>
    </SafeAreaView>);
}

function SectionTitle({ icon, title, colors }) {
    return (
      <View style={styles.sectionTitleRow}>
        <Ionicons name={icon} size={18} color={colors.brandAlt} />
        <Text style={[styles.cardTitle, { color: colors.textPrimary }]}>{title}</Text>
      </View>
    );
}

function InfoLine({ icon, label, value, tone, colors }) {
    return (
      <View style={[styles.infoLine, { borderColor: colors.border }]}>
        <Ionicons name={icon} size={18} color={tone} />
        <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{label}</Text>
        <Text style={[styles.infoValue, { color: tone }]}>{value}</Text>
      </View>
    );
}

function ReminderRow({ icon, text, color, colors }) {
    return (
      <View style={[styles.reminderRow, { backgroundColor: colors.surfaceAlt }]}>
        <Ionicons name={icon} size={18} color={color} />
        <Text style={[styles.reminderText, { color: colors.textPrimary }]}>{text}</Text>
      </View>
    );
}

function TrainingRow({ item, colors }) {
    return (
      <View style={[styles.trainingRow, { borderTopColor: colors.border }]}>
        <View style={[styles.trainingIcon, { backgroundColor: colors.surfaceAlt }]}>
          <Ionicons name={item.icon} size={18} color={colors.brandAlt} />
        </View>
        <View style={styles.trainingCopy}>
          <Text style={[styles.trainingName, { color: colors.textPrimary }]}>{item.name}</Text>
          <Text style={[styles.trainingDetail, { color: colors.textSecondary }]}>
            {item.sets} sets | {item.reps}
          </Text>
        </View>
        <Text style={[styles.intensityBadge, { color: colors.textPrimary, borderColor: colors.border }]}>
          {item.intensity}
        </Text>
      </View>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { padding: 20, gap: 14, paddingBottom: 110 },
    heroCard: { gap: 14 },
    heroTop: { flexDirection: 'row', alignItems: 'center', gap: 12 },
    heroCopy: { flex: 1, minWidth: 0 },
    heroLabel: { fontSize: 12, fontWeight: '800', textTransform: 'uppercase' },
    heroTitle: { fontSize: 20, lineHeight: 25, fontWeight: '900', marginTop: 2 },
    weekBadge: {
        width: 64,
        minHeight: 56,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 6,
    },
    weekValue: { fontSize: 17, fontWeight: '900' },
    weekLabel: { fontSize: 11, fontWeight: '800' },
    progressTrack: { height: 9, borderRadius: 8, overflow: 'hidden' },
    progressFill: { height: '100%', borderRadius: 8 },
    heroActions: { flexDirection: 'row', gap: 10 },
    heroButton: { flex: 1 },
    metricGrid: { gap: 10 },
    metricRow: { flexDirection: 'row', gap: 10 },
    sectionCard: { gap: 12 },
    sectionTitleRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    cardTitle: { fontWeight: '900', fontSize: 17 },
    sectionNote: { lineHeight: 20, fontWeight: '700' },
    infoLine: {
        minHeight: 46,
        borderWidth: 1,
        borderRadius: 8,
        paddingHorizontal: 12,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 9,
    },
    infoLabel: { flex: 1, fontSize: 13, fontWeight: '700' },
    infoValue: { fontSize: 14, fontWeight: '900' },
    reminderRow: {
        minHeight: 44,
        borderRadius: 8,
        paddingHorizontal: 12,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 9,
    },
    reminderText: { flex: 1, fontSize: 14, lineHeight: 19, fontWeight: '700' },
    trainingRow: {
        borderTopWidth: StyleSheet.hairlineWidth,
        paddingTop: 10,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 10,
    },
    trainingIcon: {
        width: 36,
        height: 36,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    trainingCopy: { flex: 1, minWidth: 0 },
    trainingName: { fontWeight: '900' },
    trainingDetail: { fontSize: 13, marginTop: 2 },
    intensityBadge: {
        borderWidth: 1,
        borderRadius: 8,
        paddingHorizontal: 8,
        paddingVertical: 5,
        fontSize: 11,
        fontWeight: '900',
    },
});
