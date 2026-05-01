import { useRouter } from 'expo-router';
import React from 'react';
import { SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Card } from '@/src/components/Card';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
function buildSuggestedPlan(input) {
    const { targetWeightLeft, currentBodyFat, targetBodyFat } = input;
    const bodyFatGap = currentBodyFat !== null && targetBodyFat !== null ? currentBodyFat - targetBodyFat : 0;
    const needsCut = (targetWeightLeft !== null && targetWeightLeft > 1) || bodyFatGap > 1.5;
    const needsGain = targetWeightLeft !== null && targetWeightLeft < -1;
    if (needsCut) {
        return {
            focus: 'Fat loss + endurance',
            weeklySessions: 5,
            plan: [
                { name: 'Circuit Squat', sets: 4, reps: '15', intensity: 'High' },
                { name: 'Push-up', sets: 4, reps: '12-15', intensity: 'Moderate' },
                { name: 'High Knees', sets: 5, reps: '40 sec', intensity: 'High' },
            ],
        };
    }
    if (needsGain) {
        return {
            focus: 'Strength + muscle gain',
            weeklySessions: 4,
            plan: [
                { name: 'Squat', sets: 5, reps: '8-10', intensity: 'High' },
                { name: 'Lunge', sets: 4, reps: '10 each leg', intensity: 'Moderate' },
                { name: 'Plank', sets: 4, reps: '45 sec', intensity: 'Moderate' },
            ],
        };
    }
    return {
        focus: 'Recomposition + consistency',
        weeklySessions: 4,
        plan: [
            { name: 'Squat', sets: 4, reps: '12', intensity: 'Moderate' },
            { name: 'Push-up', sets: 4, reps: '10-12', intensity: 'Moderate' },
            { name: 'Jumping Jack', sets: 4, reps: '30 sec', intensity: 'Low' },
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
        <Text style={[styles.title, { color: colors.textPrimary }]}>Welcome back</Text>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
          Hi {userLabel}, ready for today&apos;s session?
        </Text>

        <View style={styles.statsRow}>
          <Card style={styles.statCard}>
            <Text style={[styles.statLabel, { color: colors.textSecondary }]}>Total sessions</Text>
            <Text style={[styles.statValue, { color: colors.brandAlt }]}>{history.length}</Text>
          </Card>
          <Card style={styles.statCard}>
            <Text style={[styles.statLabel, { color: colors.textSecondary }]}>Latest score</Text>
            <Text style={[styles.statValue, { color: colors.success }]}>{recentScore}</Text>
          </Card>
          <Card style={styles.statCard}>
            <Text style={[styles.statLabel, { color: colors.textSecondary }]}>Latest reps</Text>
            <Text style={[styles.statValue, { color: colors.warning }]}>{recentReps}</Text>
          </Card>
        </View>

        <Card>
          <Text style={[styles.cardTitle, { color: colors.textPrimary }]}>Today&apos;s focus</Text>
          <Text style={[styles.cardText, { color: colors.textSecondary }]}>
            Start a guided workout with live form feedback and track your progress over time.
          </Text>
        </Card>

        <Card style={styles.goalsCard}>
          <Text style={[styles.cardTitle, { color: colors.textPrimary }]}>Goals snapshot</Text>
          <Text style={[styles.goalLine, { color: colors.textSecondary }]}>
            Target weight left:{' '}
            <Text style={[styles.goalValue, { color: targetWeightLeft !== null && targetWeightLeft > 0 ? colors.warning : colors.success }]}>
              {targetWeightLeft === null ? '--' : `${targetWeightLeft > 0 ? '+' : ''}${targetWeightLeft} kg`}
            </Text>
          </Text>
          <Text style={[styles.goalLine, { color: colors.textSecondary }]}>
            Weekly goal progress:{' '}
            <Text style={[styles.goalValue, { color: recentSessions >= weeklyTarget ? colors.success : colors.brandAlt }]}>
              {recentSessions}/{weeklyTarget} sessions
            </Text>
          </Text>
          <PrimaryButton title="Update goals" variant="secondary" onPress={() => router.push('/(app)/goals')}/>
        </Card>

        <Card style={styles.remindersCard}>
          <Text style={[styles.cardTitle, { color: colors.textPrimary }]}>Reminders</Text>
          {shouldRemindBodyUpdate ? (<Text style={[styles.reminderText, { color: colors.warning }]}>
              - Time to update your weight/body fat check-in.
            </Text>) : (<Text style={[styles.reminderTextSuccess, { color: colors.success }]}>
              - Body data is up to date this week.
            </Text>)}
          {shouldRemindWorkout ? (<Text style={[styles.reminderText, { color: colors.warning }]}>
              - No workouts logged yet. Start your first session today.
            </Text>) : (<Text style={[styles.reminderTextSuccess, { color: colors.success }]}>
              - Great consistency. Keep your streak alive.
            </Text>)}
          <PrimaryButton title="Update body data" variant="secondary" onPress={() => router.push('/(app)/body-progress')}/>
        </Card>

        <Card style={styles.trainingCard}>
          <Text style={[styles.cardTitle, { color: colors.textPrimary }]}>Suggested training (goal-based)</Text>
          <Text style={[styles.trainingMeta, { color: colors.textSecondary }]}>
            Focus: <Text style={[styles.trainingMetaStrong, { color: colors.brandAlt }]}>{suggestion.focus}</Text>
          </Text>
          <Text style={[styles.trainingMeta, { color: colors.textSecondary }]}>
            Suggested weekly sessions:{' '}
            <Text style={[styles.trainingMetaStrong, { color: colors.brandAlt }]}>{suggestion.weeklySessions}</Text>
          </Text>
          {suggestion.plan.map((item) => (<View key={item.name} style={[styles.trainingRow, { borderTopColor: colors.border }]}>
              <Text style={[styles.trainingName, { color: colors.textPrimary }]}>{item.name}</Text>
              <Text style={[styles.trainingDetail, { color: colors.textSecondary }]}>
                {item.sets} sets • {item.reps} • {item.intensity}
              </Text>
            </View>))}
          <PrimaryButton title="Start now" onPress={() => router.push('/(app)/workout-setup')}/>
        </Card>

        <PrimaryButton title="Start workout" onPress={() => router.push('/(app)/workout-setup')}/>
        <PrimaryButton title="View history" variant="secondary" onPress={() => router.push('/(app)/history')}/>
      </ScrollView>
    </SafeAreaView>);
}
const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { padding: 20, gap: 14, paddingBottom: 110 },
    title: { fontSize: 30, fontWeight: '700', marginTop: 10 },
    subtitle: { marginBottom: 8, textTransform: 'capitalize' },
    statsRow: { flexDirection: 'row', gap: 8 },
    statCard: { flex: 1, gap: 6, padding: 12 },
    statLabel: { fontSize: 11, textTransform: 'uppercase' },
    statValue: { fontSize: 20, fontWeight: '700' },
    cardTitle: { fontWeight: '700', fontSize: 16, marginBottom: 6 },
    cardText: { lineHeight: 20 },
    goalsCard: { gap: 10 },
    goalLine: { fontSize: 14 },
    goalValue: { fontWeight: '700' },
    remindersCard: { gap: 8 },
    reminderText: { lineHeight: 20 },
    reminderTextSuccess: { lineHeight: 20 },
    trainingCard: { gap: 10 },
    trainingMeta: {},
    trainingMetaStrong: { fontWeight: '700' },
    trainingRow: {
        borderTopWidth: StyleSheet.hairlineWidth,
        paddingTop: 8,
        gap: 2,
    },
    trainingName: { fontWeight: '700' },
    trainingDetail: { fontSize: 13 },
});
