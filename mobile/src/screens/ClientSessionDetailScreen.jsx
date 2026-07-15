import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams } from 'expo-router';
import React, { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card } from '@/src/components/Card';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { getExerciseMetadata } from '@/src/constants/exercise-metadata';
import { getClientSessionDetail } from '@/src/services/api/trainer';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

const SEVERITY_TONES = { high: 'danger', medium: 'warning', low: 'success' };

function formatDuration(seconds) {
    if (!seconds) return '0s';
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
}

export function ClientSessionDetailScreen() {
    const { clientId, sessionId, clientName } = useLocalSearchParams();
    const { themeMode, ensureFreshAccessToken } = useAppState();
    const colors = getThemeColors(themeMode);
    const [detail, setDetail] = useState(null);
    const [error, setError] = useState(null);

    const load = useCallback(async () => {
        try {
            setDetail(await getClientSessionDetail(clientId, sessionId));
        } catch (err) {
            if (err?.status === 401 && (await ensureFreshAccessToken())) {
                try {
                    setDetail(await getClientSessionDetail(clientId, sessionId));
                } catch (retryErr) {
                    setError(retryErr?.message || 'Failed to load session.');
                }
            } else {
                setError(err?.message || 'Failed to load session.');
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [clientId, sessionId]);

    useEffect(() => {
        load();
    }, [load]);

    const severityColor = (severity) => {
        const tone = SEVERITY_TONES[severity] ?? 'warning';
        return colors[tone] ?? colors.danger;
    };

    if (error) {
        return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
          <View style={styles.container}>
            <ScreenHeader showBack icon="barbell-outline" title="Session detail" subtitle={clientName || ''} />
            <Card style={styles.emptyCard}>
              <Ionicons name="cloud-offline-outline" size={28} color={colors.danger} />
              <Text style={[styles.centerText, { color: colors.danger }]}>{error}</Text>
            </Card>
          </View>
        </SafeAreaView>);
    }

    if (!detail) {
        return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
          <View style={[styles.container, styles.loading]}>
            <ActivityIndicator color={colors.brandAlt} size="large" />
          </View>
        </SafeAreaView>);
    }

    const meta = getExerciseMetadata(detail.exerciseId);
    const isHold = meta.measurementType === 'hold';
    const feedback = detail.feedback;
    const repStats = detail.repStats;
    const targetTotal = (detail.targetSets || 1) * (detail.targetReps || 1);
    const compliance = targetTotal > 0 ? Math.round(((detail.totalReps || 0) / targetTotal) * 100) : null;

    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>
        <ScreenHeader
          showBack
          icon="barbell-outline"
          title={detail.exerciseName}
          subtitle={`${clientName || 'Client'} · ${detail.endedAt ? new Date(detail.endedAt).toLocaleString() : ''}`}
        />

        {/* Headline numbers */}
        <View style={styles.tileRow}>
          <StatTile
            icon="repeat-outline"
            label={isHold ? 'Seconds held' : 'Total reps'}
            value={detail.totalReps ?? 0}
            colors={colors}
            tint={colors.brandAlt}
          />
          <StatTile
            icon="timer-outline"
            label="Duration"
            value={formatDuration(detail.durationSeconds)}
            colors={colors}
            tint={colors.success}
          />
        </View>
        <View style={styles.tileRow}>
          <StatTile
            icon="ribbon-outline"
            label="Form score"
            value={detail.avgFormScore != null ? `${Math.round(detail.avgFormScore)}/100` : '--'}
            colors={colors}
            tint={colors.warning ?? colors.brandAlt}
          />
          <StatTile
            icon="speedometer-outline"
            label="Difficulty"
            value={detail.difficulty}
            colors={colors}
            tint={colors.textSecondary}
          />
        </View>

        {/* Plan compliance */}
        <Card style={styles.sectionCard}>
          <SectionTitle icon="flag-outline" title="Plan vs actual" colors={colors} />
          <InfoRow label="Target" value={`${detail.targetSets} × ${detail.targetReps} ${isHold ? 'sec' : 'reps'}`} colors={colors} />
          <InfoRow label="Completed" value={`${detail.totalReps ?? 0} ${isHold ? 'sec' : 'reps'}`} colors={colors} />
          {compliance != null ? (
            <InfoRow
              label="Compliance"
              value={`${compliance}%`}
              valueColor={compliance >= 90 ? colors.success : compliance >= 60 ? (colors.warning ?? colors.brandAlt) : colors.danger}
              colors={colors}
            />
          ) : null}
          {detail.usesExternalLoad ? (
            <>
              <InfoRow label="External load" value={`${detail.externalLoadKg} kg`} colors={colors} />
              {detail.estimatedVolumeLoadKg != null ? (
                <InfoRow label="Volume load" value={`${detail.estimatedVolumeLoadKg} kg (load × reps)`} colors={colors} />
              ) : null}
            </>
          ) : null}
        </Card>

        {/* Movement quality from per-rep data */}
        {repStats && repStats.repCount > 0 ? (
          <Card style={styles.sectionCard}>
            <SectionTitle icon="pulse-outline" title="Movement quality" colors={colors} />
            <InfoRow label="Tracked reps" value={`${repStats.repCount}`} colors={colors} />
            {repStats.avgRangeOfMotion != null ? (
              <InfoRow label="Avg range of motion" value={`${repStats.avgRangeOfMotion}°`} colors={colors} />
            ) : null}
            {repStats.avgRepDurationMs != null ? (
              <InfoRow label="Avg rep tempo" value={`${(repStats.avgRepDurationMs / 1000).toFixed(1)}s / rep`} colors={colors} />
            ) : null}
            {repStats.avgFormScore != null ? (
              <InfoRow label="Avg rep form score" value={`${repStats.avgFormScore}`} colors={colors} />
            ) : null}
            {repStats.depthQualityCounts && Object.keys(repStats.depthQualityCounts).length > 0 ? (
              <View style={styles.depthBlock}>
                <Text style={[styles.depthTitle, { color: colors.textSecondary }]}>Depth quality</Text>
                {Object.entries(repStats.depthQualityCounts).map(([quality, count]) => (
                  <View key={quality} style={styles.depthRow}>
                    <Text style={[styles.depthLabel, { color: colors.textPrimary }]}>{quality.replace(/_/g, ' ')}</Text>
                    <View style={[styles.depthTrack, { backgroundColor: colors.surfaceAlt }]}>
                      <View
                        style={[
                          styles.depthFill,
                          {
                            backgroundColor: colors.brandAlt,
                            width: `${Math.min(100, Math.round((count / repStats.repCount) * 100))}%`,
                          },
                        ]}
                      />
                    </View>
                    <Text style={[styles.depthCount, { color: colors.textSecondary }]}>{count}</Text>
                  </View>
                ))}
              </View>
            ) : null}
          </Card>
        ) : null}

        {/* AI coach review */}
        {feedback ? (
          <>
            <Card style={styles.sectionCard}>
              <SectionTitle icon="analytics-outline" title="Form review" colors={colors} />
              <View style={styles.ratingRow}>
                <Text style={[styles.ratingValue, { color: colors.brandAlt }]}>{feedback.overallRating}</Text>
                <Text style={[styles.ratingOutOf, { color: colors.textSecondary }]}>/100 session rating</Text>
              </View>
              <Text style={[styles.bodyText, { color: colors.textPrimary }]}>{feedback.summaryText}</Text>
            </Card>

            {feedback.topErrors?.length > 0 ? (
              <Card style={styles.sectionCard}>
                <SectionTitle icon="warning-outline" title={`Errors detected (${feedback.errorsCount})`} colors={colors} />
                {feedback.topErrors.map((err) => (
                  <View key={err.code} style={[styles.errorRow, { borderBottomColor: colors.border }]}>
                    <View style={[styles.severityDot, { backgroundColor: severityColor(err.severity) }]} />
                    <Text style={[styles.errorLabel, { color: colors.textPrimary }]}>{err.label}</Text>
                    <Text style={[styles.errorCount, { color: colors.textSecondary }]}>×{err.count}</Text>
                    <Text style={[styles.errorSeverity, { color: severityColor(err.severity) }]}>{err.severity}</Text>
                  </View>
                ))}
              </Card>
            ) : null}

            {feedback.actionItems?.length > 0 ? (
              <Card style={styles.sectionCard}>
                <SectionTitle icon="construct-outline" title="Coaching actions" colors={colors} />
                {feedback.actionItems.map((action) => (
                  <View key={`${action.priority}-${action.title}`} style={[styles.actionBlock, { borderColor: colors.border }]}>
                    <View style={styles.actionHeader}>
                      <View style={[styles.priorityBadge, { backgroundColor: `${colors.brandAlt}1A`, borderColor: colors.brandAlt }]}>
                        <Text style={[styles.priorityText, { color: colors.brandAlt }]}>P{action.priority}</Text>
                      </View>
                      <Text style={[styles.actionTitle, { color: colors.textPrimary }]}>{action.title}</Text>
                    </View>
                    <Text style={[styles.actionText, { color: colors.textSecondary }]}>{action.why}</Text>
                    <Text style={[styles.actionText, { color: colors.textPrimary }]}>{action.howToFix}</Text>
                    {action.cue ? (
                      <View style={[styles.cueBox, { backgroundColor: colors.surfaceAlt }]}>
                        <Ionicons name="megaphone-outline" size={14} color={colors.brandAlt} />
                        <Text style={[styles.cueText, { color: colors.textPrimary }]}>&ldquo;{action.cue}&rdquo;</Text>
                      </View>
                    ) : null}
                  </View>
                ))}
              </Card>
            ) : null}
          </>
        ) : (
          <Card style={styles.emptyCard}>
            <Ionicons name="analytics-outline" size={26} color={colors.textSecondary} />
            <Text style={[styles.centerText, { color: colors.textSecondary }]}>
              No form review was generated for this session.
            </Text>
          </Card>
        )}
      </ScrollView>
    </SafeAreaView>);
}

function StatTile({ icon, label, value, colors, tint }) {
    return (
      <Card style={styles.statTile}>
        <Ionicons name={icon} size={18} color={tint} />
        <Text style={[styles.statValue, { color: colors.textPrimary }]} numberOfLines={1}>{value}</Text>
        <Text style={[styles.statLabel, { color: colors.textSecondary }]}>{label}</Text>
      </Card>
    );
}

function SectionTitle({ icon, title, colors }) {
    return (
      <View style={styles.sectionTitleRow}>
        <Ionicons name={icon} size={17} color={colors.brandAlt} />
        <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>{title}</Text>
      </View>
    );
}

function InfoRow({ label, value, colors, valueColor }) {
    return (
      <View style={[styles.infoRow, { borderBottomColor: colors.border }]}>
        <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>{label}</Text>
        <Text style={[styles.infoValue, { color: valueColor ?? colors.textPrimary }]}>{value}</Text>
      </View>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { padding: 20, gap: 14, paddingBottom: 40 },
    loading: { flex: 1, alignItems: 'center', justifyContent: 'center' },
    tileRow: { flexDirection: 'row', gap: 10 },
    statTile: { flex: 1, alignItems: 'center', gap: 4, paddingVertical: 14 },
    statValue: { fontSize: 18, fontWeight: '900', textTransform: 'capitalize' },
    statLabel: { fontSize: 11, fontWeight: '800' },
    sectionCard: { gap: 10 },
    sectionTitleRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    sectionTitle: { fontSize: 16, fontWeight: '900' },
    infoRow: {
        minHeight: 38,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottomWidth: StyleSheet.hairlineWidth,
    },
    infoLabel: { fontWeight: '700', fontSize: 13 },
    infoValue: { fontWeight: '900', fontSize: 13, textTransform: 'capitalize' },
    ratingRow: { flexDirection: 'row', alignItems: 'baseline', gap: 4 },
    ratingValue: { fontSize: 34, fontWeight: '900' },
    ratingOutOf: { fontSize: 13, fontWeight: '700' },
    bodyText: { fontSize: 14, lineHeight: 21 },
    errorRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
        minHeight: 40,
        borderBottomWidth: StyleSheet.hairlineWidth,
    },
    severityDot: { width: 9, height: 9, borderRadius: 5 },
    errorLabel: { flex: 1, fontWeight: '800', fontSize: 13 },
    errorCount: { fontWeight: '900', fontSize: 13 },
    errorSeverity: { fontSize: 11, fontWeight: '900', textTransform: 'uppercase', width: 58, textAlign: 'right' },
    actionBlock: { borderWidth: 1, borderRadius: 10, padding: 12, gap: 6 },
    actionHeader: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    priorityBadge: { borderWidth: 1, borderRadius: 6, paddingHorizontal: 7, paddingVertical: 2 },
    priorityText: { fontSize: 11, fontWeight: '900' },
    actionTitle: { flex: 1, fontWeight: '900', fontSize: 14 },
    actionText: { fontSize: 13, lineHeight: 19 },
    cueBox: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
        borderRadius: 8,
        padding: 10,
        marginTop: 2,
    },
    cueText: { flex: 1, fontSize: 13, fontWeight: '800', fontStyle: 'italic' },
    depthBlock: { gap: 6, marginTop: 4 },
    depthTitle: { fontSize: 12, fontWeight: '800' },
    depthRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    depthLabel: { width: 90, fontSize: 12, fontWeight: '700', textTransform: 'capitalize' },
    depthTrack: { flex: 1, height: 8, borderRadius: 4, overflow: 'hidden' },
    depthFill: { height: 8, borderRadius: 4 },
    depthCount: { width: 24, fontSize: 12, fontWeight: '900', textAlign: 'right' },
    emptyCard: { alignItems: 'center', gap: 10, paddingVertical: 24 },
    centerText: { textAlign: 'center', lineHeight: 20 },
});
