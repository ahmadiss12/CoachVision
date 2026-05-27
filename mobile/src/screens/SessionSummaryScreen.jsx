import { useRouter } from 'expo-router';
import React from 'react';
import { SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Card } from '../components/Card';
import { PrimaryButton } from '../components/PrimaryButton';
import { getExerciseMetadata, supportsExternalLoad } from '../constants/exercise-metadata';
import {
    buildSuggestedRepTarget,
    fatigueLabel,
} from '../services/fatigue-plan';
import { useAppState } from '../state/app-state';
import { colors } from '../theme/colors';

function severityColor(severity) {
    if (severity === 'high') {
        return colors.danger;
    }
    if (severity === 'medium') {
        return colors.warning;
    }
    return colors.success;
}

function IssueItem({ issue, breakdown }) {
    const detail = breakdown?.[issue.code] ?? {};
    return (
      <View style={styles.issueItem}>
        <View style={styles.issueHeader}>
          <View style={[styles.severityDot, { backgroundColor: severityColor(issue.severity) }]} />
          <Text style={styles.issueTitle}>{issue.label}</Text>
          <Text style={styles.issueCount}>{issue.count}x</Text>
        </View>
        {detail.why ? <Text style={styles.issueText}>{detail.why}</Text> : null}
        {detail.fix ? <Text style={styles.fixText}>Fix: {detail.fix}</Text> : null}
      </View>
    );
}

function ActionItem({ item }) {
    return (
      <View style={styles.actionItem}>
        <Text style={styles.actionNumber}>{item.priority}</Text>
        <View style={styles.actionCopy}>
          <Text style={styles.actionTitle}>{item.title}</Text>
          <Text style={styles.actionText}>{item.howToFix || item.cue}</Text>
        </View>
      </View>
    );
}

function fatigueColor(level) {
    if (level === 'high') {
        return colors.danger;
    }
    if (level === 'moderate') {
        return colors.warning;
    }
    return colors.success;
}

function FatigueOutlook({ prediction, summary }) {
    const exerciseId = summary?.exerciseId ?? summary?.exerciseName;
    const meta = getExerciseMetadata(exerciseId);
    const unitName = meta.measurementType === 'hold' ? 'seconds' : 'reps';
    const loadKg = Number(
        prediction?.featureSnapshot?.self_report?.externalLoadKg
        ?? summary?.externalLoadKg
        ?? 0
    );
    if (!prediction) {
        return (
          <Card style={styles.card}>
            <Text style={styles.sectionTitle}>Next-session outlook</Text>
            <Text style={styles.emptyText}>
              No fatigue prediction is available yet. Run a fatigue check before the next workout.
            </Text>
          </Card>
        );
    }

    const plan = buildSuggestedRepTarget({
        prediction,
        lastSession: summary,
        fallbackReps: summary?.reps || (meta.measurementType === 'hold' ? 30 : 10),
        exerciseId,
    });
    const factors = prediction.explainability?.slice(0, 3) ?? [];
    const snapshot = prediction.featureSnapshot ?? {};
    return (
      <Card style={styles.card}>
        <Text style={styles.sectionTitle}>Next-session outlook</Text>
        <View style={styles.outlookRow}>
          <View style={styles.outlookBox}>
            <Text style={[styles.outlookValue, { color: colors.brandAlt }]}>
              {prediction.readinessScore}
            </Text>
            <Text style={styles.statLabel}>Readiness</Text>
          </View>
          <View style={styles.outlookBox}>
            <Text style={[styles.outlookValueSmall, { color: fatigueColor(prediction.fatigueLevel) }]}>
              {fatigueLabel(prediction.fatigueLevel)}
            </Text>
            <Text style={styles.statLabel}>Fatigue</Text>
          </View>
          <View style={styles.outlookBox}>
            <Text style={[styles.outlookValue, { color: colors.success }]}>
              {plan.targetReps}
            </Text>
            <Text style={styles.statLabel}>Next {unitName}</Text>
          </View>
        </View>
        <Text style={styles.note}>{prediction.recommendation}</Text>
        <View style={styles.loadMiniRow}>
          <Text style={styles.loadMiniText}>Recent {unitName}: {snapshot.reps_7d ?? 0}</Text>
          <Text style={styles.loadMiniText}>Recent sessions: {snapshot.sessions_7d ?? 0}</Text>
          {supportsExternalLoad(exerciseId) ? (
            <Text style={styles.loadMiniText}>Load: {loadKg} kg</Text>
          ) : null}
        </View>
        {factors.length ? (
          <View style={styles.factorList}>
            {factors.map((factor) => (
              <View key={factor.key} style={styles.factorItem}>
                <Text style={styles.factorTitle}>
                  {factor.label} ({factor.impact > 0 ? '+' : ''}{factor.impact})
                </Text>
                <Text style={styles.factorText}>{factor.detail}</Text>
              </View>
            ))}
          </View>
        ) : null}
      </Card>
    );
}

function MistakeSummary({ feedback }) {
    if (!feedback) {
        return (
          <Card style={styles.card}>
            <Text style={styles.sectionTitle}>Form review</Text>
            <Text style={styles.emptyText}>
              No detailed form review was generated for this session.
            </Text>
          </Card>
        );
    }

    const hasIssues = feedback.topErrors?.length > 0;
    return (
      <Card style={styles.card}>
        <Text style={styles.sectionTitle}>What to improve</Text>
        <Text style={styles.note}>{feedback.summaryText}</Text>

        {hasIssues ? (
          <View style={styles.issueList}>
            {feedback.topErrors.map((issue) => (
              <IssueItem
                key={issue.code}
                issue={issue}
                breakdown={feedback.errorBreakdown}
              />
            ))}
          </View>
        ) : (
          <Text style={styles.successText}>No major mistakes were detected in this workout.</Text>
        )}

        {feedback.actionItems?.length ? (
          <View style={styles.actionList}>
            <Text style={styles.subTitle}>Next workout cues</Text>
            {feedback.actionItems.map((item) => (
              <ActionItem key={`${item.priority}-${item.title}`} item={item} />
            ))}
          </View>
        ) : null}
      </Card>
    );
}

export function SessionSummaryScreen() {
    const router = useRouter();
    const { latestSummary, latestFeedback, latestFatiguePrediction } = useAppState();
    const fatiguePrediction = latestSummary?.fatiguePrediction ?? latestFatiguePrediction;
    const summaryMeta = getExerciseMetadata(latestSummary?.exerciseId ?? latestSummary?.exerciseName);
    const primaryStatLabel = summaryMeta.measurementType === 'hold' ? 'Seconds' : 'Reps';
    const exerciseLabel = summaryMeta.label || latestSummary?.exerciseName;

    return (
      <SafeAreaView style={styles.safe}>
        <ScrollView contentContainerStyle={styles.container}>
          <Text style={styles.title}>Session summary</Text>
          {!latestSummary ? (
            <Card>
              <Text style={styles.label}>No summary available yet.</Text>
            </Card>
          ) : (
            <>
              <Card style={styles.card}>
                <Text style={styles.label}>Exercise: {exerciseLabel}</Text>
                <Text style={styles.label}>Difficulty: {latestSummary.difficulty}</Text>
                <View style={styles.statRow}>
                  <View style={styles.statBox}>
                    <Text style={styles.statValue}>{latestSummary.reps}</Text>
                    <Text style={styles.statLabel}>{primaryStatLabel}</Text>
                  </View>
                  <View style={styles.statBox}>
                    <Text style={styles.statValue}>{latestSummary.durationSec}s</Text>
                    <Text style={styles.statLabel}>Duration</Text>
                  </View>
                  <View style={styles.statBox}>
                    <Text style={styles.statValue}>{latestSummary.score}</Text>
                    <Text style={styles.statLabel}>Score</Text>
                  </View>
                </View>
                <Text style={styles.note}>Coach note: {latestSummary.notes}</Text>
              </Card>

              <FatigueOutlook prediction={fatiguePrediction} summary={latestSummary} />
              <MistakeSummary feedback={latestFeedback ?? latestSummary.feedback} />
            </>
          )}

          <PrimaryButton title="Back to home" onPress={() => router.replace('/(app)/(tabs)')} />
          <PrimaryButton title="View history" variant="secondary" onPress={() => router.push('/(app)/history')} />
        </ScrollView>
      </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: colors.background },
    container: { padding: 20, gap: 12, paddingBottom: 28 },
    title: { color: colors.textPrimary, fontSize: 28, fontWeight: '700', marginTop: 10 },
    card: { gap: 10 },
    label: { color: colors.textPrimary, fontSize: 15 },
    note: { color: colors.textSecondary, marginTop: 4, lineHeight: 20 },
    sectionTitle: { color: colors.textPrimary, fontSize: 18, fontWeight: '800' },
    subTitle: { color: colors.textPrimary, fontSize: 15, fontWeight: '800' },
    emptyText: { color: colors.textSecondary, lineHeight: 20 },
    successText: { color: colors.success, fontWeight: '700', lineHeight: 20 },
    statRow: { flexDirection: 'row', gap: 8, marginTop: 4 },
    statBox: {
        flex: 1,
        minHeight: 64,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 8,
        backgroundColor: colors.surfaceAlt,
        borderWidth: 1,
        borderColor: colors.border,
    },
    statValue: { color: colors.textPrimary, fontSize: 20, fontWeight: '900' },
    statLabel: { color: colors.textSecondary, fontSize: 11, fontWeight: '700', marginTop: 2 },
    outlookRow: { flexDirection: 'row', gap: 8 },
    outlookBox: {
        flex: 1,
        minHeight: 68,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 8,
        backgroundColor: colors.surfaceAlt,
        borderWidth: 1,
        borderColor: colors.border,
        paddingHorizontal: 6,
    },
    outlookValue: { fontSize: 21, fontWeight: '900' },
    outlookValueSmall: { fontSize: 13, fontWeight: '900', textAlign: 'center' },
    loadMiniRow: { flexDirection: 'row', gap: 8, flexWrap: 'wrap' },
    loadMiniText: { color: colors.textSecondary, fontSize: 12, fontWeight: '700' },
    factorList: { gap: 8 },
    factorItem: {
        borderRadius: 8,
        backgroundColor: colors.surfaceAlt,
        borderWidth: 1,
        borderColor: colors.border,
        padding: 10,
        gap: 3,
    },
    factorTitle: { color: colors.textPrimary, fontSize: 13, fontWeight: '800' },
    factorText: { color: colors.textSecondary, lineHeight: 18 },
    issueList: { gap: 8, marginTop: 4 },
    issueItem: {
        borderRadius: 8,
        backgroundColor: colors.surfaceAlt,
        borderWidth: 1,
        borderColor: colors.border,
        padding: 10,
        gap: 6,
    },
    issueHeader: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    severityDot: { width: 9, height: 9, borderRadius: 999 },
    issueTitle: { flex: 1, color: colors.textPrimary, fontSize: 14, fontWeight: '800' },
    issueCount: { color: colors.textSecondary, fontSize: 12, fontWeight: '800' },
    issueText: { color: colors.textSecondary, lineHeight: 18 },
    fixText: { color: colors.textPrimary, lineHeight: 18, fontWeight: '600' },
    actionList: { gap: 8, marginTop: 6 },
    actionItem: { flexDirection: 'row', gap: 10, alignItems: 'flex-start' },
    actionNumber: {
        width: 24,
        height: 24,
        borderRadius: 12,
        overflow: 'hidden',
        textAlign: 'center',
        lineHeight: 24,
        color: '#fff',
        backgroundColor: colors.brandAlt,
        fontWeight: '900',
    },
    actionCopy: { flex: 1 },
    actionTitle: { color: colors.textPrimary, fontWeight: '800' },
    actionText: { color: colors.textSecondary, lineHeight: 18, marginTop: 2 },
});
