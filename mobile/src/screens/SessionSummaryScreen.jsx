import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Card } from '../components/Card';
import { MetricTile } from '../components/MetricTile';
import { PrimaryButton } from '../components/PrimaryButton';
import { ScreenHeader } from '../components/ScreenHeader';
import { getExerciseMetadata, supportsExternalLoad } from '../constants/exercise-metadata';
import {
    buildSuggestedRepTarget,
    fatigueLabel,
} from '../services/fatigue-plan';
import { useAppState } from '../state/app-state';
import { getThemeColors } from '../theme/colors';

function severityColor(severity, colors) {
    if (severity === 'high') {
        return colors.danger;
    }
    if (severity === 'medium') {
        return colors.warning;
    }
    return colors.success;
}

function IssueItem({ issue, breakdown, colors }) {
    const [isOpen, setIsOpen] = useState(false);
    const detail = breakdown?.[issue.code] ?? {};
    const tone = severityColor(issue.severity, colors);
    const hasDetails = Boolean(detail.why || detail.fix);
    return (
      <View style={[styles.issueItem, { backgroundColor: colors.surfaceAlt, borderColor: colors.border }]}>
        <View style={styles.issueHeader}>
          <View style={[styles.severityIcon, { backgroundColor: `${tone}22` }]}>
            <Ionicons name={issue.severity === 'high' ? 'warning-outline' : 'information-circle-outline'} size={18} color={tone} />
          </View>
          <View style={styles.issueCopy}>
            <Text style={[styles.issueTitle, { color: colors.textPrimary }]}>{issue.label}</Text>
            <Text style={[styles.issueMeta, { color: colors.textSecondary }]}>
              {issue.count}x | {issue.severity || 'low'} priority
            </Text>
          </View>
        </View>

        {hasDetails ? (
          <Pressable
            onPress={() => setIsOpen((prev) => !prev)}
            style={({ pressed }) => [
              styles.detailsButton,
              { borderColor: colors.border, opacity: pressed ? 0.82 : 1 },
            ]}
          >
            <Ionicons name={isOpen ? 'chevron-up' : 'chevron-down'} size={17} color={colors.brandAlt} />
            <Text style={[styles.detailsButtonText, { color: colors.brandAlt }]}>
              {isOpen ? 'Hide details' : 'View details'}
            </Text>
          </Pressable>
        ) : null}

        {isOpen ? (
          <View style={[styles.issueDetails, { borderTopColor: colors.border }]}>
            {detail.why ? (
              <DetailLine icon="help-circle-outline" label="Why it matters" text={detail.why} colors={colors} />
            ) : null}
            {detail.fix ? (
              <DetailLine icon="construct-outline" label="Fix" text={detail.fix} colors={colors} />
            ) : null}
          </View>
        ) : null}
      </View>
    );
}

function DetailLine({ icon, label, text, colors }) {
    return (
      <View style={styles.detailLine}>
        <Ionicons name={icon} size={16} color={colors.brandAlt} />
        <View style={styles.detailCopy}>
          <Text style={[styles.detailLabel, { color: colors.textPrimary }]}>{label}</Text>
          <Text style={[styles.detailText, { color: colors.textSecondary }]}>{text}</Text>
        </View>
      </View>
    );
}

function ActionItem({ item, colors }) {
    return (
      <View style={[styles.actionItem, { backgroundColor: colors.surfaceAlt }]}>
        <View style={[styles.actionNumber, { backgroundColor: colors.brand }]}>
          <Text style={styles.actionNumberText}>{item.priority}</Text>
        </View>
        <View style={styles.actionCopy}>
          <Text style={[styles.actionTitle, { color: colors.textPrimary }]}>{item.title}</Text>
          <Text style={[styles.actionText, { color: colors.textSecondary }]}>{item.howToFix || item.cue}</Text>
        </View>
      </View>
    );
}

function fatigueColor(level, colors) {
    if (level === 'high') {
        return colors.danger;
    }
    if (level === 'moderate') {
        return colors.warning;
    }
    return colors.success;
}

function FatigueOutlook({ prediction, summary, colors }) {
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
            <SectionTitle icon="pulse-outline" title="Next-session outlook" colors={colors} />
            <EmptyLine icon="analytics-outline" text="No fatigue prediction is available yet." colors={colors} />
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
        <SectionTitle icon="pulse-outline" title="Next-session outlook" colors={colors} />
        <View style={styles.outlookRow}>
          <OutlookBox icon="battery-charging-outline" label="Readiness" value={prediction.readinessScore} color={colors.brandAlt} colors={colors} />
          <OutlookBox icon="warning-outline" label="Fatigue" value={fatigueLabel(prediction.fatigueLevel)} color={fatigueColor(prediction.fatigueLevel, colors)} colors={colors} compact />
          <OutlookBox icon="flag-outline" label={`Next ${unitName}`} value={plan.targetReps} color={colors.success} colors={colors} />
        </View>
        <Text style={[styles.note, { color: colors.textSecondary }]}>{prediction.recommendation}</Text>
        <View style={styles.loadMiniRow}>
          <MiniPill icon="repeat-outline" text={`Recent ${unitName}: ${snapshot.reps_7d ?? 0}`} colors={colors} />
          <MiniPill icon="calendar-outline" text={`Sessions: ${snapshot.sessions_7d ?? 0}`} colors={colors} />
          {supportsExternalLoad(exerciseId) ? (
            <MiniPill icon="barbell-outline" text={`Load: ${loadKg} kg`} colors={colors} />
          ) : null}
        </View>
        {factors.length ? (
          <View style={styles.factorList}>
            {factors.map((factor) => (
              <View key={factor.key} style={[styles.factorItem, { backgroundColor: colors.surfaceAlt, borderColor: colors.border }]}>
                <Text style={[styles.factorTitle, { color: colors.textPrimary }]}>
                  {factor.label} ({factor.impact > 0 ? '+' : ''}{factor.impact})
                </Text>
                <Text style={[styles.factorText, { color: colors.textSecondary }]}>{factor.detail}</Text>
              </View>
            ))}
          </View>
        ) : null}
      </Card>
    );
}

function OutlookBox({ icon, label, value, color, colors, compact = false }) {
    return (
      <View style={[styles.outlookBox, { backgroundColor: colors.surfaceAlt, borderColor: colors.border }]}>
        <Ionicons name={icon} size={17} color={color} />
        <Text style={[compact ? styles.outlookValueSmall : styles.outlookValue, { color }]} numberOfLines={2}>
          {value}
        </Text>
        <Text style={[styles.statLabel, { color: colors.textSecondary }]}>{label}</Text>
      </View>
    );
}

function MiniPill({ icon, text, colors }) {
    return (
      <View style={[styles.miniPill, { borderColor: colors.border }]}>
        <Ionicons name={icon} size={14} color={colors.textSecondary} />
        <Text style={[styles.loadMiniText, { color: colors.textSecondary }]}>{text}</Text>
      </View>
    );
}

function MistakeSummary({ feedback, colors }) {
    if (!feedback) {
        return (
          <Card style={styles.card}>
            <SectionTitle icon="clipboard-outline" title="Form review" colors={colors} />
            <EmptyLine icon="document-text-outline" text="No detailed form review was generated for this session." colors={colors} />
          </Card>
        );
    }

    const hasIssues = feedback.topErrors?.length > 0;
    return (
      <Card style={styles.card}>
        <SectionTitle icon="clipboard-check-outline" title="Form review" colors={colors} />
        <Text style={[styles.note, { color: colors.textSecondary }]}>{feedback.summaryText}</Text>

        {hasIssues ? (
          <View style={styles.issueList}>
            {feedback.topErrors.map((issue) => (
              <IssueItem
                key={issue.code}
                issue={issue}
                breakdown={feedback.errorBreakdown}
                colors={colors}
              />
            ))}
          </View>
        ) : (
          <View style={[styles.successBox, { backgroundColor: `${colors.success}18` }]}>
            <Ionicons name="checkmark-circle-outline" size={18} color={colors.success} />
            <Text style={[styles.successText, { color: colors.success }]}>No major mistakes were detected.</Text>
          </View>
        )}

        {feedback.actionItems?.length ? (
          <View style={styles.actionList}>
            <SectionTitle icon="bulb-outline" title="Next workout cues" colors={colors} small />
            {feedback.actionItems.map((item) => (
              <ActionItem key={`${item.priority}-${item.title}`} item={item} colors={colors} />
            ))}
          </View>
        ) : null}
      </Card>
    );
}

function SectionTitle({ icon, title, colors, small = false }) {
    return (
      <View style={styles.sectionTitleRow}>
        <Ionicons name={icon} size={small ? 16 : 19} color={colors.brandAlt} />
        <Text style={[small ? styles.subTitle : styles.sectionTitle, { color: colors.textPrimary }]}>{title}</Text>
      </View>
    );
}

function EmptyLine({ icon, text, colors }) {
    return (
      <View style={[styles.emptyLine, { backgroundColor: colors.surfaceAlt }]}>
        <Ionicons name={icon} size={18} color={colors.textSecondary} />
        <Text style={[styles.emptyText, { color: colors.textSecondary }]}>{text}</Text>
      </View>
    );
}

export function SessionSummaryScreen() {
    const router = useRouter();
    const { latestSummary, latestFeedback, latestFatiguePrediction, themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const fatiguePrediction = latestSummary?.fatiguePrediction ?? latestFatiguePrediction;
    const summaryMeta = getExerciseMetadata(latestSummary?.exerciseId ?? latestSummary?.exerciseName);
    const primaryStatLabel = summaryMeta.measurementType === 'hold' ? 'Seconds' : 'Reps';
    const exerciseLabel = summaryMeta.label || latestSummary?.exerciseName;

    return (
      <SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
        <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>
          <ScreenHeader
            showBack
            icon="checkmark-done-circle-outline"
            title="Session summary"
            subtitle={latestSummary ? `${exerciseLabel} | ${latestSummary.difficulty || 'Workout'}` : 'Finish a workout to see your summary.'}
          />
          {!latestSummary ? (
            <Card style={styles.card}>
              <EmptyLine icon="barbell-outline" text="No summary available yet." colors={colors} />
            </Card>
          ) : (
            <>
              <Card style={styles.heroCard}>
                <View style={styles.summaryTop}>
                  <View style={[styles.exerciseIcon, { backgroundColor: `${colors.brandAlt}22` }]}>
                    <Ionicons name="barbell-outline" size={24} color={colors.brandAlt} />
                  </View>
                  <View style={styles.summaryCopy}>
                    <Text style={[styles.exerciseTitle, { color: colors.textPrimary }]}>{exerciseLabel}</Text>
                    <Text style={[styles.exerciseMeta, { color: colors.textSecondary }]}>Difficulty: {latestSummary.difficulty || '--'}</Text>
                  </View>
                </View>
                <View style={styles.metricRow}>
                  <MetricTile icon="repeat-outline" label={primaryStatLabel} value={latestSummary.reps} tone="brand" />
                  <MetricTile icon="timer-outline" label="Duration" value={`${latestSummary.durationSec}s`} tone="info" />
                  <MetricTile icon="trophy-outline" label="Score" value={latestSummary.score} tone="success" />
                </View>
                <View style={[styles.coachNote, { backgroundColor: colors.surfaceAlt }]}>
                  <Ionicons name="chatbubble-ellipses-outline" size={18} color={colors.brandAlt} />
                  <Text style={[styles.note, { color: colors.textSecondary }]}>Coach note: {latestSummary.notes}</Text>
                </View>
              </Card>

              <FatigueOutlook prediction={fatiguePrediction} summary={latestSummary} colors={colors} />
              <MistakeSummary feedback={latestFeedback ?? latestSummary.feedback} colors={colors} />
            </>
          )}

          <PrimaryButton icon="home-outline" title="Back to home" onPress={() => router.replace('/(app)/(tabs)')} />
          <PrimaryButton icon="time-outline" title="View history" variant="secondary" onPress={() => router.push('/(app)/history')} />
        </ScrollView>
      </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { padding: 20, gap: 12, paddingBottom: 32 },
    card: { gap: 12 },
    heroCard: { gap: 14 },
    summaryTop: { flexDirection: 'row', alignItems: 'center', gap: 12 },
    exerciseIcon: {
        width: 48,
        height: 48,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    summaryCopy: { flex: 1, minWidth: 0 },
    exerciseTitle: { fontSize: 18, fontWeight: '900' },
    exerciseMeta: { marginTop: 2, fontWeight: '700' },
    metricRow: { flexDirection: 'row', gap: 8 },
    coachNote: {
        borderRadius: 8,
        padding: 12,
        flexDirection: 'row',
        alignItems: 'flex-start',
        gap: 8,
    },
    note: { flex: 1, marginTop: 0, lineHeight: 20, fontWeight: '700' },
    sectionTitleRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    sectionTitle: { fontSize: 18, fontWeight: '900' },
    subTitle: { fontSize: 15, fontWeight: '900' },
    emptyLine: {
        minHeight: 48,
        borderRadius: 8,
        paddingHorizontal: 12,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 9,
    },
    emptyText: { flex: 1, lineHeight: 20, fontWeight: '700' },
    successBox: {
        borderRadius: 8,
        padding: 10,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    successText: { flex: 1, fontWeight: '900', lineHeight: 20 },
    statLabel: { fontSize: 11, fontWeight: '800', marginTop: 2, textAlign: 'center' },
    outlookRow: { flexDirection: 'row', gap: 8 },
    outlookBox: {
        flex: 1,
        minHeight: 84,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 8,
        backgroundColor: 'transparent',
        borderWidth: 1,
        paddingHorizontal: 6,
        gap: 3,
    },
    outlookValue: { fontSize: 22, fontWeight: '900' },
    outlookValueSmall: { fontSize: 13, fontWeight: '900', textAlign: 'center' },
    loadMiniRow: { flexDirection: 'row', gap: 8, flexWrap: 'wrap' },
    miniPill: {
        minHeight: 32,
        borderWidth: 1,
        borderRadius: 8,
        paddingHorizontal: 9,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
    },
    loadMiniText: { fontSize: 12, fontWeight: '800' },
    factorList: { gap: 8 },
    factorItem: {
        borderRadius: 8,
        borderWidth: 1,
        padding: 10,
        gap: 3,
    },
    factorTitle: { fontSize: 13, fontWeight: '900' },
    factorText: { lineHeight: 18 },
    issueList: { gap: 8, marginTop: 2 },
    issueItem: {
        borderRadius: 8,
        borderWidth: 1,
        padding: 10,
        gap: 9,
    },
    issueHeader: { flexDirection: 'row', alignItems: 'center', gap: 10 },
    severityIcon: {
        width: 36,
        height: 36,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    issueCopy: { flex: 1, minWidth: 0 },
    issueTitle: { fontSize: 14, fontWeight: '900' },
    issueMeta: { fontSize: 12, fontWeight: '800', marginTop: 2, textTransform: 'capitalize' },
    detailsButton: {
        alignSelf: 'flex-start',
        minHeight: 34,
        borderWidth: 1,
        borderRadius: 8,
        paddingHorizontal: 10,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
    },
    detailsButtonText: { fontSize: 13, fontWeight: '900' },
    issueDetails: {
        borderTopWidth: StyleSheet.hairlineWidth,
        paddingTop: 10,
        gap: 10,
    },
    detailLine: { flexDirection: 'row', gap: 8, alignItems: 'flex-start' },
    detailCopy: { flex: 1, minWidth: 0 },
    detailLabel: { fontSize: 13, fontWeight: '900' },
    detailText: { marginTop: 2, lineHeight: 18 },
    actionList: { gap: 8, marginTop: 4 },
    actionItem: {
        flexDirection: 'row',
        gap: 10,
        alignItems: 'flex-start',
        borderRadius: 8,
        padding: 10,
    },
    actionNumber: {
        width: 28,
        height: 28,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    actionNumberText: {
        color: '#fff',
        fontWeight: '900',
    },
    actionCopy: { flex: 1, minWidth: 0 },
    actionTitle: { fontWeight: '900' },
    actionText: { lineHeight: 18, marginTop: 2 },
});
