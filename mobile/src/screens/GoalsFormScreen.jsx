import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Card } from '@/src/components/Card';
import { Input } from '@/src/components/Input';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

function toNumber(text) {
    const value = Number(text);
    return Number.isFinite(value) ? value : 0;
}

export function GoalsFormScreen({ mode }) {
    const router = useRouter();
    const { goals, profile, saveGoals, themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const [targetWeightKg, setTargetWeightKg] = useState(goals?.targetWeightKg ? String(goals.targetWeightKg) : profile?.weightKg ? String(profile.weightKg) : '');
    const [targetBodyFatPercent, setTargetBodyFatPercent] = useState(goals?.targetBodyFatPercent
        ? String(goals.targetBodyFatPercent)
        : profile?.bodyFatPercent
            ? String(profile.bodyFatPercent)
            : '');
    const [weeklyWorkoutTarget, setWeeklyWorkoutTarget] = useState(goals?.weeklyWorkoutTarget ? String(goals.weeklyWorkoutTarget) : '4');
    const onSave = () => {
        saveGoals({
            targetWeightKg: toNumber(targetWeightKg),
            targetBodyFatPercent: toNumber(targetBodyFatPercent),
            weeklyWorkoutTarget: toNumber(weeklyWorkoutTarget),
        });
        if (mode === 'onboarding') {
            router.replace('/(app)/(tabs)');
        }
        else {
            router.back();
        }
    };
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <ScreenHeader
          icon="flag-outline"
          title={mode === 'onboarding' ? 'Set your goals' : 'Edit goals'}
          subtitle={mode === 'onboarding'
            ? 'Targets help CoachVision tune each session.'
            : 'Adjust targets as your training changes.'}
        />

        <Card style={styles.formCard}>
          <Input icon="scale-outline" label="Target weight (kg)" value={targetWeightKg} keyboardType="numeric" placeholder="e.g. 68" onChangeText={setTargetWeightKg}/>
          <Input icon="water-outline" label="Target body fat (%)" value={targetBodyFatPercent} keyboardType="numeric" placeholder="e.g. 16" onChangeText={setTargetBodyFatPercent}/>
          <Input icon="calendar-number-outline" label="Weekly workout target" value={weeklyWorkoutTarget} keyboardType="numeric" placeholder="e.g. 4" onChangeText={setWeeklyWorkoutTarget}/>
        </Card>

        <Card style={styles.previewCard}>
          <View style={[styles.previewIcon, { backgroundColor: colors.surfaceAlt }]}>
            <Ionicons name="sparkles-outline" size={22} color={colors.brandAlt} />
          </View>
          <View style={styles.previewCopy}>
            <Text style={[styles.previewTitle, { color: colors.textPrimary }]}>Goal profile ready</Text>
            <Text style={[styles.previewText, { color: colors.textSecondary }]}>
              Weekly target: {weeklyWorkoutTarget || '--'} sessions
            </Text>
          </View>
        </Card>

        <PrimaryButton
          icon="checkmark-circle-outline"
          title={mode === 'onboarding' ? 'Save goals and continue' : 'Save goal changes'}
          onPress={onSave}
        />
      </ScrollView>
    </SafeAreaView>);
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { flexGrow: 1, padding: 20, gap: 14, paddingBottom: 32 },
    formCard: { gap: 12 },
    previewCard: { flexDirection: 'row', alignItems: 'center', gap: 12 },
    previewIcon: {
        width: 46,
        height: 46,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    previewCopy: { flex: 1, minWidth: 0 },
    previewTitle: { fontSize: 16, fontWeight: '900' },
    previewText: { marginTop: 2, lineHeight: 19 },
});
