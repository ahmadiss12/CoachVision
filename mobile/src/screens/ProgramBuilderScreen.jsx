import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card } from '@/src/components/Card';
import { Input } from '@/src/components/Input';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { createProgram } from '@/src/services/api/programs';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

const EXERCISE_CATALOG = [
    { id: 'squat', label: 'Squat' },
    { id: 'pushup', label: 'Push-up' },
    { id: 'lunge', label: 'Lunge' },
    { id: 'deadlift', label: 'Deadlift' },
    { id: 'bicep_curl', label: 'Bicep curl' },
    { id: 'shoulder_press', label: 'Shoulder press' },
    { id: 'situp', label: 'Sit-up' },
    { id: 'jumping_jack', label: 'Jumping jack' },
    { id: 'high_knees', label: 'High knees' },
    { id: 'mountain_climber', label: 'Mountain climber' },
    { id: 'plank', label: 'Plank (sec)' },
    { id: 'wall_sit', label: 'Wall sit (sec)' },
];

const LOAD_BEARING = new Set(['squat', 'lunge', 'deadlift', 'bicep_curl', 'shoulder_press']);

function Stepper({ label, value, onChange, min = 1, max = 99, step = 1, colors }) {
    return (
      <View style={styles.stepper}>
        <Text style={[styles.stepperLabel, { color: colors.textSecondary }]}>{label}</Text>
        <View style={styles.stepperControls}>
          <Pressable
            onPress={() => onChange(Math.max(min, value - step))}
            style={[styles.stepperButton, { borderColor: colors.border }]}
          >
            <Ionicons name="remove" size={16} color={colors.textPrimary} />
          </Pressable>
          <Text style={[styles.stepperValue, { color: colors.textPrimary }]}>{value}</Text>
          <Pressable
            onPress={() => onChange(Math.min(max, value + step))}
            style={[styles.stepperButton, { borderColor: colors.border }]}
          >
            <Ionicons name="add" size={16} color={colors.textPrimary} />
          </Pressable>
        </View>
      </View>
    );
}

export function ProgramBuilderScreen() {
    const router = useRouter();
    const { themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const [name, setName] = useState('');
    const [weeks, setWeeks] = useState(1);
    const [days, setDays] = useState([]);
    const [isSaving, setIsSaving] = useState(false);
    const [pickingForDay, setPickingForDay] = useState(null);

    const addDay = () => {
        const used = new Set(days.map((d) => d.dayIndex));
        let nextIndex = 1;
        while (used.has(nextIndex) && nextIndex <= weeks * 7) {
            nextIndex += 1;
        }
        if (nextIndex > weeks * 7) {
            Alert.alert('Cycle full', `A ${weeks}-week program has at most ${weeks * 7} days.`);
            return;
        }
        setDays((prev) => [...prev, { dayIndex: nextIndex, exercises: [] }]);
    };

    const removeDay = (dayIndex) => {
        setDays((prev) => prev.filter((d) => d.dayIndex !== dayIndex));
    };

    const addExercise = (dayIndex, exerciseId) => {
        setDays((prev) =>
            prev.map((d) =>
                d.dayIndex === dayIndex
                    ? {
                          ...d,
                          exercises: [
                              ...d.exercises,
                              {
                                  exerciseId,
                                  targetSets: 3,
                                  targetReps: 10,
                                  restSeconds: 60,
                                  externalLoadKg: 0,
                              },
                          ],
                      }
                    : d,
            ),
        );
        setPickingForDay(null);
    };

    const updateExercise = (dayIndex, position, patch) => {
        setDays((prev) =>
            prev.map((d) =>
                d.dayIndex === dayIndex
                    ? {
                          ...d,
                          exercises: d.exercises.map((e, i) => (i === position ? { ...e, ...patch } : e)),
                      }
                    : d,
            ),
        );
    };

    const removeExercise = (dayIndex, position) => {
        setDays((prev) =>
            prev.map((d) =>
                d.dayIndex === dayIndex
                    ? { ...d, exercises: d.exercises.filter((_, i) => i !== position) }
                    : d,
            ),
        );
    };

    const save = async () => {
        if (!name.trim()) {
            Alert.alert('Missing name', 'Give the program a name.');
            return;
        }
        const trainingDays = days.filter((d) => d.exercises.length > 0);
        if (trainingDays.length === 0) {
            Alert.alert('Empty program', 'Add at least one day with an exercise.');
            return;
        }
        setIsSaving(true);
        try {
            await createProgram({
                name: name.trim(),
                weeks,
                days: trainingDays.map((d) => ({
                    dayIndex: d.dayIndex,
                    exercises: d.exercises.map((e) => ({
                        exerciseId: e.exerciseId,
                        targetSets: e.targetSets,
                        targetReps: e.targetReps,
                        restSeconds: e.restSeconds,
                        externalLoadKg: LOAD_BEARING.has(e.exerciseId) && e.externalLoadKg > 0 ? e.externalLoadKg : null,
                    })),
                })),
            });
            router.back();
        } catch (err) {
            Alert.alert('Save failed', err?.message || 'Could not save the program.');
        } finally {
            setIsSaving(false);
        }
    };

    const labelFor = (exerciseId) =>
        EXERCISE_CATALOG.find((e) => e.id === exerciseId)?.label ?? exerciseId;

    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <ScreenHeader showBack icon="construct-outline" title="New program" subtitle="Days repeat weekly across the cycle." />

        <Card style={styles.formCard}>
          <Input icon="text-outline" label="Program name" value={name} placeholder="e.g. Beginner full body" onChangeText={setName} />
          <Stepper label={`Cycle length: ${weeks} week${weeks > 1 ? 's' : ''}`} value={weeks} onChange={setWeeks} min={1} max={4} colors={colors} />
        </Card>

        {days.map((day) => (
          <Card key={day.dayIndex} style={styles.dayCard}>
            <View style={styles.dayHeader}>
              <Text style={[styles.dayTitle, { color: colors.textPrimary }]}>Day {day.dayIndex}</Text>
              <Pressable hitSlop={8} onPress={() => removeDay(day.dayIndex)}>
                <Ionicons name="trash-outline" size={18} color={colors.danger} />
              </Pressable>
            </View>

            {day.exercises.map((exercise, position) => (
              <View key={`${exercise.exerciseId}-${position}`} style={[styles.exerciseBlock, { borderColor: colors.border }]}>
                <View style={styles.exerciseHeader}>
                  <Text style={[styles.exerciseName, { color: colors.brandAlt }]}>{labelFor(exercise.exerciseId)}</Text>
                  <Pressable hitSlop={8} onPress={() => removeExercise(day.dayIndex, position)}>
                    <Ionicons name="close-circle-outline" size={19} color={colors.textSecondary} />
                  </Pressable>
                </View>
                <Stepper
                  label="Sets"
                  value={exercise.targetSets}
                  onChange={(v) => updateExercise(day.dayIndex, position, { targetSets: v })}
                  min={1}
                  max={10}
                  colors={colors}
                />
                <Stepper
                  label={exercise.exerciseId === 'plank' || exercise.exerciseId === 'wall_sit' ? 'Hold (sec)' : 'Reps'}
                  value={exercise.targetReps}
                  onChange={(v) => updateExercise(day.dayIndex, position, { targetReps: v })}
                  min={1}
                  max={200}
                  step={exercise.exerciseId === 'plank' || exercise.exerciseId === 'wall_sit' ? 5 : 1}
                  colors={colors}
                />
                <Stepper
                  label="Rest (sec)"
                  value={exercise.restSeconds}
                  onChange={(v) => updateExercise(day.dayIndex, position, { restSeconds: v })}
                  min={0}
                  max={300}
                  step={15}
                  colors={colors}
                />
                {LOAD_BEARING.has(exercise.exerciseId) ? (
                  <Stepper
                    label="Load (kg)"
                    value={exercise.externalLoadKg}
                    onChange={(v) => updateExercise(day.dayIndex, position, { externalLoadKg: v })}
                    min={0}
                    max={200}
                    step={2.5}
                    colors={colors}
                  />
                ) : null}
              </View>
            ))}

            {pickingForDay === day.dayIndex ? (
              <View style={styles.pickerWrap}>
                {EXERCISE_CATALOG.map((entry) => (
                  <Pressable
                    key={entry.id}
                    onPress={() => addExercise(day.dayIndex, entry.id)}
                    style={[styles.pickerChip, { borderColor: colors.border, backgroundColor: colors.surfaceAlt }]}
                  >
                    <Text style={[styles.pickerChipText, { color: colors.textPrimary }]}>{entry.label}</Text>
                  </Pressable>
                ))}
              </View>
            ) : (
              <PrimaryButton
                icon="add-outline"
                title="Add exercise"
                variant="secondary"
                onPress={() => setPickingForDay(day.dayIndex)}
              />
            )}
          </Card>
        ))}

        <PrimaryButton icon="calendar-outline" title="Add training day" variant="secondary" onPress={addDay} />
        <PrimaryButton
          icon="save-outline"
          title={isSaving ? 'Saving…' : 'Save program'}
          onPress={save}
          disabled={isSaving}
        />
      </ScrollView>
    </SafeAreaView>);
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { padding: 20, gap: 14, paddingBottom: 40 },
    formCard: { gap: 14 },
    dayCard: { gap: 10 },
    dayHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
    dayTitle: { fontSize: 16, fontWeight: '900' },
    exerciseBlock: { borderWidth: 1, borderRadius: 10, padding: 12, gap: 8 },
    exerciseHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
    exerciseName: { fontWeight: '900', fontSize: 14 },
    stepper: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
    stepperLabel: { fontSize: 13, fontWeight: '700', flex: 1 },
    stepperControls: { flexDirection: 'row', alignItems: 'center', gap: 10 },
    stepperButton: {
        width: 30,
        height: 30,
        borderRadius: 8,
        borderWidth: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    stepperValue: { minWidth: 40, textAlign: 'center', fontWeight: '900', fontSize: 14 },
    pickerWrap: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
    pickerChip: {
        borderWidth: 1,
        borderRadius: 999,
        paddingHorizontal: 12,
        paddingVertical: 7,
    },
    pickerChipText: { fontSize: 12, fontWeight: '800' },
});
