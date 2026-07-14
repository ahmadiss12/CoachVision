import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useEffect, useState } from 'react';
import { FlatList, Pressable, RefreshControl, SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { Card } from '@/src/components/Card';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { getExerciseMetadata } from '@/src/constants/exercise-metadata';
import { listClientSessions } from '@/src/services/api/trainer';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

function formatExerciseName(exerciseId) {
    const meta = getExerciseMetadata(exerciseId);
    if (meta.label) {
        return meta.label;
    }
    return String(exerciseId || 'Workout')
        .split(/[_-]+/)
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' ');
}

export function ClientSessionsScreen() {
    const router = useRouter();
    const { clientId, clientName } = useLocalSearchParams();
    const { themeMode, ensureFreshAccessToken } = useAppState();
    const colors = getThemeColors(themeMode);
    const [sessions, setSessions] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const refresh = useCallback(async () => {
        if (!clientId) {
            return;
        }
        setIsLoading(true);
        setError(null);
        try {
            setSessions(await listClientSessions(clientId));
        } catch (err) {
            if (err?.status === 401 && (await ensureFreshAccessToken())) {
                try {
                    setSessions(await listClientSessions(clientId));
                } catch (retryErr) {
                    setError(retryErr?.message || 'Failed to load sessions.');
                }
            } else {
                setError(err?.message || 'Failed to load sessions.');
            }
        } finally {
            setIsLoading(false);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [clientId]);

    useEffect(() => {
        refresh();
    }, [refresh]);

    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <View style={styles.container}>
        <ScreenHeader
          showBack
          icon="barbell-outline"
          title={clientName ? `${clientName}'s workouts` : 'Client workouts'}
          subtitle={`${sessions.length} completed ${sessions.length === 1 ? 'session' : 'sessions'}`}
        />
        {error ? (
          <Card style={styles.emptyCard}>
            <Ionicons name="cloud-offline-outline" size={28} color={colors.danger} />
            <Text style={[styles.empty, { color: colors.danger }]}>{error}</Text>
          </Card>
        ) : null}
        {!error && sessions.length === 0 && !isLoading ? (
          <Card style={styles.emptyCard}>
            <View style={[styles.emptyIcon, { backgroundColor: colors.surfaceAlt }]}>
              <Ionicons name="calendar-clear-outline" size={28} color={colors.textSecondary} />
            </View>
            <Text style={[styles.emptyTitle, { color: colors.textPrimary }]}>No workouts yet</Text>
            <Text style={[styles.empty, { color: colors.textSecondary }]}>
              Completed sessions will appear here once your client works out.
            </Text>
          </Card>
        ) : (
          <FlatList
            data={sessions}
            keyExtractor={(item) => item.id}
            refreshControl={<RefreshControl refreshing={isLoading} onRefresh={refresh} tintColor={colors.brandAlt} />}
            contentContainerStyle={styles.list}
            showsVerticalScrollIndicator={false}
            renderItem={({ item }) => {
                const meta = getExerciseMetadata(item.exerciseId);
                const isHold = meta.measurementType === 'hold';
                return (
                  <Pressable
                    onPress={() =>
                      router.push({
                        pathname: '/(app)/client-session-detail',
                        params: { clientId, sessionId: item.id, clientName },
                      })
                    }
                  >
                  <Card style={styles.item}>
                    <View style={styles.itemHeader}>
                      <View style={[styles.itemIcon, { backgroundColor: `${colors.brandAlt}22` }]}>
                        <Ionicons name="barbell-outline" size={20} color={colors.brandAlt} />
                      </View>
                      <View style={styles.itemCopy}>
                        <Text style={[styles.itemTitle, { color: colors.textPrimary }]}>
                          {formatExerciseName(item.exerciseId)}
                        </Text>
                        <Text style={[styles.itemSub, { color: colors.textSecondary }]}>
                          {item.endedAt ? new Date(item.endedAt).toLocaleString() : '--'}
                        </Text>
                      </View>
                      <View style={[styles.scorePill, { backgroundColor: colors.surfaceAlt }]}>
                        <Text style={[styles.scoreValue, { color: colors.success }]}>
                          {item.avgFormScore != null ? Math.round(item.avgFormScore) : '--'}
                        </Text>
                        <Text style={[styles.scoreLabel, { color: colors.textSecondary }]}>score</Text>
                      </View>
                    </View>
                    <View style={styles.statRow}>
                      <MiniStat
                        icon="repeat-outline"
                        label={isHold ? 'sec' : 'reps'}
                        value={item.totalReps ?? 0}
                        colors={colors}
                      />
                      <MiniStat
                        icon="timer-outline"
                        label="time"
                        value={`${item.durationSeconds ?? 0}s`}
                        colors={colors}
                      />
                      <MiniStat icon="speedometer-outline" label="level" value={item.difficulty || '--'} colors={colors} />
                    </View>
                    <View style={styles.detailHint}>
                      <Text style={[styles.detailHintText, { color: colors.brandAlt }]}>View full breakdown</Text>
                      <Ionicons name="chevron-forward" size={14} color={colors.brandAlt} />
                    </View>
                  </Card>
                  </Pressable>
                );
            }}
          />
        )}
      </View>
    </SafeAreaView>);
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
    itemSub: { fontSize: 12, marginTop: 2 },
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
    detailHint: { flexDirection: 'row', alignItems: 'center', justifyContent: 'flex-end', gap: 2 },
    detailHintText: { fontSize: 12, fontWeight: '900' },
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
});
