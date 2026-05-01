import { useRouter } from 'expo-router';
import React from 'react';
import { SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { Card } from '../components/Card';
import { PrimaryButton } from '../components/PrimaryButton';
import { useAppState } from '../state/app-state';
import { colors } from '../theme/colors';
export function SessionSummaryScreen() {
    const router = useRouter();
    const { latestSummary, latestFeedback } = useAppState();
    return (<SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Text style={styles.title}>Session summary</Text>
        {!latestSummary ? (<Card>
            <Text style={styles.label}>No summary available yet.</Text>
          </Card>) : (<Card style={styles.card}>
            <Text style={styles.label}>Exercise: {latestSummary.exerciseName}</Text>
            <Text style={styles.label}>Difficulty: {latestSummary.difficulty}</Text>
            <Text style={styles.label}>Reps: {latestSummary.reps}</Text>
            <Text style={styles.label}>Duration: {latestSummary.durationSec}s</Text>
            <Text style={styles.label}>Score: {latestSummary.score}/100</Text>
            <Text style={styles.note}>Coach note: {latestSummary.notes}</Text>
            {latestFeedback?.topErrors?.length ? (<Text style={styles.label}>
                Top issue: {latestFeedback.topErrors[0].label} ({latestFeedback.topErrors[0].count})
              </Text>) : null}
            {latestFeedback?.actionItems?.length ? (<Text style={styles.note}>
                Next step: {latestFeedback.actionItems[0].title}
              </Text>) : null}
          </Card>)}

        <PrimaryButton title="Back to home" onPress={() => router.replace('/(app)/(tabs)')}/>
        <PrimaryButton title="View history" variant="secondary" onPress={() => router.push('/(app)/history')}/>
      </View>
    </SafeAreaView>);
}
const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: colors.background },
    container: { flex: 1, padding: 20, gap: 12 },
    title: { color: colors.textPrimary, fontSize: 28, fontWeight: '700', marginTop: 10 },
    card: { gap: 8 },
    label: { color: colors.textPrimary, fontSize: 15 },
    note: { color: colors.textSecondary, marginTop: 4 },
});
