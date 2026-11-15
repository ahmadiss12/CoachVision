import { Ionicons } from '@expo/vector-icons';
import React, { useCallback, useEffect, useState } from 'react';
import { Alert, FlatList, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Card } from '@/src/components/Card';
import { LockedNotice } from '@/src/components/LockedNotice';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { getExerciseMetadata } from '@/src/constants/exercise-metadata';
import { exportDailyReportPdf } from '@/src/services/reports/daily-report-pdf';
import { useAppState } from '@/src/state/app-state';
import { usePlan } from '@/src/state/use-plan';
import { getThemeColors } from '@/src/theme/colors';

function HistoryItem({ item, colors }) {
    const meta = getExerciseMetadata(item.exerciseId ?? item.exerciseName);
    const primaryLabel = meta.measurementType === 'hold' ? 'sec' : 'reps';
    return (<Card style={styles.item}>
      <View style={styles.itemHeader}>
        <View style={[styles.itemIcon, { backgroundColor: `${colors.brandAlt}22` }]}>
          <Ionicons name="barbell-outline" size={20} color={colors.brandAlt} />
        </View>
        <View style={styles.itemCopy}>
          <Text style={[styles.itemTitle, { color: colors.textPrimary }]}>{item.exerciseName}</Text>
          <Text style={[styles.itemDate, { color: colors.textSecondary }]}>{new Date(item.endedAt).toLocaleString()}</Text>
        </View>
        <View style={[styles.scorePill, { backgroundColor: colors.surfaceAlt }]}>
          <Text style={[styles.scoreValue, { color: colors.success }]}>{item.score ?? 0}</Text>
          <Text style={[styles.scoreLabel, { color: colors.textSecondary }]}>score</Text>
        </View>
      </View>
      <View style={styles.statRow}>
        <MiniStat icon="repeat-outline" label={primaryLabel} value={item.reps ?? 0} colors={colors} />
        <MiniStat icon="timer-outline" label="time" value={`${item.durationSec ?? 0}s`} colors={colors} />
        <MiniStat icon="speedometer-outline" label="level" value={item.difficulty || '--'} colors={colors} />
      </View>
    </Card>);
}

function MiniStat({ icon, label, value, colors }) {
    return (
      <View style={[styles.miniStat, { backgroundColor: colors.surfaceAlt }]}>
        <Ionicons name={icon} size={15} color={colors.textSecondary} />
        <Text style={[styles.miniValue, { color: colors.textPrimary }]} numberOfLines={1}>{value}</Text>
        <Text style={[styles.miniLabel, { color: colors.textSecondary }]}>{label}</Text>
      </View>
    );
}

export function HistoryScreen() {
    const router = useRouter();
    const { history, loadHistory, themeMode } = useAppState();
    const { can, limits, planForHigherLimit } = usePlan();
    const canExportPdf = can('pdf_report');
    const [isExporting, setIsExporting] = useState(false);
    const colors = getThemeColors(themeMode);
    useEffect(() => {
        loadHistory();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    const handleExportToday = useCallback(async () => {
        if (!canExportPdf) {
            router.push('/(app)/plans');
            return;
        }
        if (isExporting) {
            return;
        }
        setIsExporting(true);
        try {
            const result = await exportDailyReportPdf();
            if (!result.shared) {
                Alert.alert('PDF created', 'Sharing is not available on this device.');
            }
        } catch (error) {
            Alert.alert('Export failed', error?.message || 'Unable to create the daily coach report.');
        } finally {
            setIsExporting(false);
        }
    }, [canExportPdf, isExporting, router]);
    const historyNotice = limits.historyDays == null ? null : (
      <LockedNotice
        plan={planForHigherLimit('historyDays')}
        title={`Showing the last ${limits.historyDays} days`}
        message="Older workouts are saved and come back when you upgrade."
      />
    );
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <View style={styles.container}>
        <ScreenHeader
          showBack
          icon="time-outline"
          title="Workout history"
          subtitle={`${history.length} saved ${history.length === 1 ? 'session' : 'sessions'}`}
          right={
            <Pressable
              disabled={isExporting}
              onPress={handleExportToday}
              style={({ pressed }) => [
                styles.exportButton,
                {
                  backgroundColor: colors.surfaceAlt,
                  borderColor: colors.border,
                  opacity: isExporting ? 0.5 : pressed ? 0.82 : 1,
                },
              ]}
            >
              <Ionicons
                name={!canExportPdf ? 'lock-closed' : isExporting ? 'sync-outline' : 'document-text-outline'}
                size={17}
                color={canExportPdf ? colors.brandAlt : colors.warning}
              />
              <Text style={[styles.exportText, { color: colors.textPrimary }]} numberOfLines={1}>
                {isExporting ? 'Exporting' : 'PDF'}
              </Text>
            </Pressable>
          }
        />
        {history.length === 0 ? historyNotice : null}
        {history.length === 0 ? (<Card style={styles.emptyCard}>
            <View style={[styles.emptyIcon, { backgroundColor: colors.surfaceAlt }]}>
              <Ionicons name="calendar-clear-outline" size={28} color={colors.textSecondary} />
            </View>
            <Text style={[styles.emptyTitle, { color: colors.textPrimary }]}>No sessions yet</Text>
            <Text style={[styles.empty, { color: colors.textSecondary }]}>Complete a workout and your summary will appear here.</Text>
          </Card>) : (<FlatList
            ListFooterComponent={historyNotice}
            data={history}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => <HistoryItem item={item} colors={colors}/>}
            contentContainerStyle={styles.list}
            showsVerticalScrollIndicator={false}
          />)}
      </View>
    </SafeAreaView>);
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { flex: 1, padding: 20, gap: 14 },
    list: { gap: 10, paddingBottom: 24 },
    item: { gap: 12 },
    itemHeader: { flexDirection: 'row', alignItems: 'center', gap: 12 },
    itemIcon: {
        width: 44,
        height: 44,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    itemCopy: { flex: 1, minWidth: 0 },
    itemTitle: { fontWeight: '900', fontSize: 16 },
    itemDate: { fontSize: 12, marginTop: 2 },
    scorePill: {
        width: 58,
        minHeight: 48,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    scoreValue: { fontSize: 18, fontWeight: '900' },
    scoreLabel: { fontSize: 10, fontWeight: '800' },
    statRow: { flexDirection: 'row', gap: 8 },
    miniStat: {
        flex: 1,
        minHeight: 58,
        borderRadius: 8,
        padding: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    miniValue: { fontSize: 13, fontWeight: '900', marginTop: 2, textTransform: 'capitalize' },
    miniLabel: { fontSize: 10, fontWeight: '800', marginTop: 1 },
    emptyCard: { alignItems: 'center', gap: 10, paddingVertical: 28 },
    emptyIcon: {
        width: 58,
        height: 58,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    emptyTitle: { fontSize: 18, fontWeight: '900' },
    empty: { textAlign: 'center', lineHeight: 20 },
    exportButton: {
        minWidth: 84,
        height: 42,
        borderRadius: 8,
        borderWidth: 1,
        paddingHorizontal: 12,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
    },
    exportText: { fontSize: 13, fontWeight: '900' },
});
