import { Ionicons } from '@expo/vector-icons';
import React, { useEffect } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card } from '@/src/components/Card';
import { PlanTag } from '@/src/components/PlanTag';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { PLAN_ORDER, planLabel } from '@/src/services/plan-access';
import { useAppState } from '@/src/state/app-state';
import { usePlan } from '@/src/state/use-plan';
import { getThemeColors } from '@/src/theme/colors';

const AUDIENCE = {
    free: 'For getting started, and for training with a coach',
    plus: 'For training on your own',
    pro: 'For trainers and coaches',
};

// Numbers come from the server's plan list when it has them, so this screen
// cannot drift from what the server enforces.
function planHighlights(plan, plans) {
    const limits = plans[plan]?.limits ?? {};
    if (plan === 'free') {
        const exerciseCount = limits.exerciseIds?.length ?? 6;
        const historyDays = limits.historyDays ?? 14;
        return [
            `Unlimited AI-tracked workouts on ${exerciseCount} bodyweight exercises`,
            'Live rep counting and voice form cues',
            'Your top issue and one cue after every workout',
            `The last ${historyDays} days of workout history`,
            'Training with a coach: programs and chat',
        ];
    }
    if (plan === 'plus') {
        return [
            'Every exercise, including weighted training',
            'The full form review after every workout',
            'Readiness score, the reasons behind it, and a next-session target',
            'Unlimited workout history',
            'Body-fat tracking',
            'Daily report as a PDF',
        ];
    }
    const clients = limits.maxClients ?? 30;
    return [
        'Everything in Plus',
        `Up to ${clients} active clients`,
        limits.maxPrograms == null ? 'Unlimited programs' : `Up to ${limits.maxPrograms} programs`,
        "The full review of every client's session",
        'Live monitoring on the web dashboard',
    ];
}

export function PlansScreen() {
    const { themeMode, loadEntitlements } = useAppState();
    const { entitlements, earlyAccess, plan: currentPlan } = usePlan();
    const colors = getThemeColors(themeMode);

    useEffect(() => {
        loadEntitlements();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return (
      <SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
        <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>
          <ScreenHeader
            showBack
            icon="ribbon-outline"
            title="Plans"
            subtitle={`You are on ${planLabel(currentPlan)}.`}
          />

          <View style={[styles.banner, { backgroundColor: colors.surfaceAlt, borderColor: colors.border }]}>
            <Ionicons
              name={earlyAccess ? 'gift-outline' : 'information-circle-outline'}
              size={20}
              color={colors.brandAlt}
            />
            <Text style={[styles.bannerText, { color: colors.textSecondary }]}>
              {earlyAccess
                ? 'Early access: everything below is free for now. You will be told in the app before that changes.'
                : 'Plus and Pro are not available to buy yet. They will appear here when they are.'}
            </Text>
          </View>

          {PLAN_ORDER.map((plan) => {
              const isCurrent = plan === currentPlan;
              return (
                <Card key={plan} style={[styles.planCard, isCurrent && { borderColor: colors.brandAlt, borderWidth: 2 }]}>
                  <View style={styles.planHeader}>
                    <View style={styles.planTitleBlock}>
                      <Text style={[styles.planName, { color: colors.textPrimary }]}>{planLabel(plan)}</Text>
                      <Text style={[styles.planAudience, { color: colors.textSecondary }]}>{AUDIENCE[plan]}</Text>
                    </View>
                    {isCurrent ? (
                      <View style={[styles.currentPill, { backgroundColor: `${colors.brandAlt}22` }]}>
                        <Text style={[styles.currentText, { color: colors.brandAlt }]}>Your plan</Text>
                      </View>
                    ) : (
                      <PlanTag plan={plan} />
                    )}
                  </View>
                  {planHighlights(plan, entitlements.plans).map((line) => (
                    <View key={line} style={styles.highlightRow}>
                      <Ionicons name="checkmark-circle" size={16} color={colors.success} />
                      <Text style={[styles.highlightText, { color: colors.textSecondary }]}>{line}</Text>
                    </View>
                  ))}
                </Card>
              );
          })}
        </ScrollView>
      </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { padding: 20, gap: 14, paddingBottom: 40 },
    banner: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        gap: 10,
        borderWidth: 1,
        borderRadius: 8,
        padding: 12,
    },
    bannerText: { flex: 1, fontSize: 13, lineHeight: 19 },
    planCard: { gap: 10 },
    planHeader: { flexDirection: 'row', alignItems: 'center', gap: 10 },
    planTitleBlock: { flex: 1, gap: 2 },
    planName: { fontSize: 20, fontWeight: '900' },
    planAudience: { fontSize: 12 },
    currentPill: { borderRadius: 999, paddingHorizontal: 10, paddingVertical: 4 },
    currentText: { fontSize: 11, fontWeight: '900' },
    highlightRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 8 },
    highlightText: { flex: 1, fontSize: 13, lineHeight: 19 },
});
