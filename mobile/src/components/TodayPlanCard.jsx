import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useCallback, useEffect, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Card } from '@/src/components/Card';
import { getTodayPlan } from '@/src/services/api/programs';
import { useAppState } from '@/src/state/app-state';

const LEVEL_TONES = { low: 'success', moderate: 'warning', high: 'danger' };

export function TodayPlanCard({ colors }) {
    const router = useRouter();
    const { ensureFreshAccessToken } = useAppState();
    const [plan, setPlan] = useState(null);

    const load = useCallback(async () => {
        try {
            setPlan(await getTodayPlan());
        } catch (err) {
            if (err?.status === 401 && (await ensureFreshAccessToken())) {
                try {
                    setPlan(await getTodayPlan());
                } catch {
                    setPlan(null);
                }
            } else {
                setPlan(null);
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        load();
    }, [load]);

    if (!plan || !plan.hasPlan) {
        return null;
    }

    const levelColor = (level) => colors[LEVEL_TONES[level] ?? 'warning'] ?? colors.brandAlt;

    return (
      <Card style={styles.card}>
        <View style={styles.headerRow}>
          <View style={[styles.headerIcon, { backgroundColor: `${colors.brandAlt}22` }]}>
            <Ionicons name="clipboard-outline" size={20} color={colors.brandAlt} />
          </View>
          <View style={styles.headerCopy}>
            <Text style={[styles.eyebrow, { color: colors.textSecondary }]}>
              Today&apos;s plan · {plan.programName}
            </Text>
            <Text style={[styles.title, { color: colors.textPrimary }]}>
              {plan.restDay ? 'Rest day' : `Day ${plan.dayIndex} of ${plan.cycleLength}`}
            </Text>
            {plan.trainerName ? (
              <Text style={[styles.coach, { color: colors.textSecondary }]}>Coach: {plan.trainerName}</Text>
            ) : null}
          </View>
        </View>

        {plan.restDay ? (
          <Text style={[styles.restText, { color: colors.textSecondary }]}>
            {plan.message || 'Recover well — no exercises scheduled today.'}
          </Text>
        ) : (
          plan.exercises.map((item) => {
            const wasAdjusted = item.adjustedReps !== item.prescribedReps || item.adjustedSets !== item.prescribedSets;
            return (
              <Pressable
                key={item.exerciseId}
                onPress={() =>
                  router.push({
                    pathname: '/(app)/workout-setup',
                    params: {
                      exerciseId: item.exerciseId,
                      targetReps: String(item.adjustedReps),
                      externalLoadKg: item.externalLoadKg != null ? String(item.externalLoadKg) : undefined,
                      assignmentId: plan.assignmentId ?? undefined,
                    },
                  })
                }
                style={[styles.exerciseRow, { borderTopColor: colors.border }]}
              >
                <View style={styles.exerciseCopy}>
                  <Text style={[styles.exerciseName, { color: colors.textPrimary }]}>{item.exerciseName}</Text>
                  <View style={styles.targetLine}>
                    {wasAdjusted ? (
                      <Text style={[styles.struck, { color: colors.textSecondary }]}>
                        {item.prescribedSets}×{item.prescribedReps}
                      </Text>
                    ) : null}
                    <Text style={[styles.target, { color: colors.textPrimary }]}>
                      {item.adjustedSets}×{item.adjustedReps}
                    </Text>
                    {item.externalLoadKg ? (
                      <Text style={[styles.load, { color: colors.textSecondary }]}>@{item.externalLoadKg}kg</Text>
                    ) : null}
                  </View>
                  {item.adjustmentNote ? (
                    <Text style={[styles.note, { color: levelColor(item.fatigueLevel) }]}>{item.adjustmentNote}</Text>
                  ) : null}
                </View>
                <View style={[styles.readinessPill, { backgroundColor: `${levelColor(item.fatigueLevel)}1A` }]}>
                  <Text style={[styles.readinessValue, { color: levelColor(item.fatigueLevel) }]}>
                    {item.readinessScore}
                  </Text>
                  <Text style={[styles.readinessLabel, { color: colors.textSecondary }]}>ready</Text>
                </View>
                <Ionicons name="play-circle-outline" size={26} color={colors.brandAlt} />
              </Pressable>
            );
          })
        )}
        {plan.dayNotes ? (
          <Text style={[styles.dayNotes, { color: colors.textSecondary }]}>{plan.dayNotes}</Text>
        ) : null}
      </Card>
    );
}

const styles = StyleSheet.create({
    card: { gap: 10 },
    headerRow: { flexDirection: 'row', alignItems: 'center', gap: 12 },
    headerIcon: {
        width: 44,
        height: 44,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    headerCopy: { flex: 1, minWidth: 0 },
    eyebrow: { fontSize: 12, fontWeight: '800', textTransform: 'uppercase' },
    title: { fontSize: 18, fontWeight: '900', marginTop: 2 },
    coach: { fontSize: 12, fontWeight: '700', marginTop: 1 },
    restText: { fontSize: 14, lineHeight: 20, fontWeight: '600' },
    exerciseRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 10,
        paddingTop: 10,
        borderTopWidth: StyleSheet.hairlineWidth,
    },
    exerciseCopy: { flex: 1, minWidth: 0 },
    exerciseName: { fontWeight: '900', fontSize: 15 },
    targetLine: { flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 2 },
    struck: { fontSize: 13, fontWeight: '700', textDecorationLine: 'line-through' },
    target: { fontSize: 14, fontWeight: '900' },
    load: { fontSize: 12, fontWeight: '800' },
    note: { fontSize: 11, fontWeight: '700', marginTop: 3, lineHeight: 15 },
    readinessPill: {
        width: 52,
        minHeight: 44,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    readinessValue: { fontSize: 16, fontWeight: '900' },
    readinessLabel: { fontSize: 9, fontWeight: '800' },
    dayNotes: { fontSize: 12, fontWeight: '600', fontStyle: 'italic' },
});
