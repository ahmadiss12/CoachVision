import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { Input } from '@/src/components/Input';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { useAppState } from '@/src/state/app-state';
import { colors } from '@/src/theme/colors';
function toNumber(text) {
    const value = Number(text);
    return Number.isFinite(value) ? value : 0;
}
export function GoalsFormScreen({ mode }) {
    const router = useRouter();
    const { goals, profile, saveGoals } = useAppState();
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
    return (<SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Text style={styles.title}>
          {mode === 'onboarding' ? 'Set your goals' : 'Edit goals'}
        </Text>
        <Text style={styles.subtitle}>
          {mode === 'onboarding'
            ? 'Tell us your target body goals so we can guide your progress.'
            : 'Update your targets anytime as your fitness improves.'}
        </Text>

        <Input label="Target weight (kg)" value={targetWeightKg} keyboardType="numeric" placeholder="e.g. 68" onChangeText={setTargetWeightKg}/>
        <Input label="Target body fat (%)" value={targetBodyFatPercent} keyboardType="numeric" placeholder="e.g. 16" onChangeText={setTargetBodyFatPercent}/>
        <Input label="Weekly workout target (sessions)" value={weeklyWorkoutTarget} keyboardType="numeric" placeholder="e.g. 4" onChangeText={setWeeklyWorkoutTarget}/>

        <PrimaryButton title={mode === 'onboarding' ? 'Save goals and continue' : 'Save goal changes'} onPress={onSave}/>
      </View>
    </SafeAreaView>);
}
const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: colors.background },
    container: { flex: 1, padding: 20, gap: 12 },
    title: { color: colors.textPrimary, fontSize: 28, fontWeight: '700', marginTop: 10 },
    subtitle: { color: colors.textSecondary, marginBottom: 8 },
});
