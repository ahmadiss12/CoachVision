import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import React, { useEffect, useMemo, useState } from 'react';
import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View, } from 'react-native';
import { PrimaryButton } from '../components/PrimaryButton';
import { getExerciseMetadata, supportsExternalLoad } from '../constants/exercise-metadata';
import {
    buildSuggestedRepTarget,
    fatigueLabel,
    findLatestSessionForExercise,
} from '../services/fatigue-plan';
import { useAppState } from '../state/app-state';
import { getThemeColors } from '../theme/colors';
const exercises = [
    { id: 'squat', label: 'Squat', icon: 'body-outline', image: require('@/images/squat.jpg') },
    { id: 'pushup', label: 'Push-up', icon: 'hand-left-outline', image: require('@/images/pushup.jpg') },
    { id: 'lunge', label: 'Lunge', icon: 'walk-outline', image: require('@/images/lunge.jpg') },
    { id: 'deadlift', label: 'Deadlift', icon: 'barbell-outline', image: require('@/images/squat.jpg') },
    { id: 'plank', label: 'Plank', icon: 'male-outline', image: require('@/images/8e1aplank.jpg') },
    { id: 'bicep_curl', label: 'Bicep curl', icon: 'barbell-outline', image: require('@/images/squat.jpg') },
    { id: 'shoulder_press', label: 'Shoulder press', icon: 'barbell-outline', image: require('@/images/squat.jpg') },
    { id: 'situp', label: 'Sit-up', icon: 'accessibility-outline', image: require('@/images/8e1aplank.jpg') },
    { id: 'jumping_jack', label: 'Jumping jack', icon: 'flash-outline', image: require('@/images/jumpingjack.jpg') },
    { id: 'high_knees', label: 'High knees', icon: 'trending-up-outline', image: require('@/images/highKnees.jpg') },
    { id: 'mountain_climber', label: 'Mountain climber', icon: 'trending-up-outline', image: require('@/images/8e1aplank.jpg') },
    { id: 'wall_sit', label: 'Wall sit', icon: 'body-outline', image: require('@/images/squat.jpg') },
];
const difficulties = [
    { id: 'beginner', label: 'Beginner', hint: 'Lighter load, more rest' },
    { id: 'intermediate', label: 'Intermediate', hint: 'Balanced challenge' },
    { id: 'advanced', label: 'Advanced', hint: 'Harder thresholds' },
];
const GRID_GAP = 12;
const H_PADDING = 20;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const chunk = (items, size) => items.reduce((rows, item, index) => {
    if (index % size === 0) {
        rows.push([]);
    }
    rows[rows.length - 1].push(item);
    return rows;
}, []);

function ReadinessStepper({ label, value, onDecrease, onIncrease, colors }) {
    return (<View style={styles.stepperRow}>
      <Text style={[styles.stepperLabel, { color: colors.textSecondary }]}>{label}</Text>
      <View style={styles.stepperControls}>
        <Pressable onPress={onDecrease} style={[styles.stepperButton, { borderColor: colors.border }]}>
          <Text style={[styles.stepperButtonText, { color: colors.textPrimary }]}>-</Text>
        </Pressable>
        <Text style={[styles.stepperValue, { color: colors.textPrimary }]}>{value}</Text>
        <Pressable onPress={onIncrease} style={[styles.stepperButton, { borderColor: colors.border }]}>
          <Text style={[styles.stepperButtonText, { color: colors.textPrimary }]}>+</Text>
        </Pressable>
      </View>
    </View>);
}

export function WorkoutSetupScreen() {
    const router = useRouter();
    const {
        startWorkout,
        loadExercises,
        availableExercises,
        latestError,
        themeMode,
        profile,
        history,
        latestFatiguePrediction,
        predictFatigue,
        loadFatigueHistory,
    } = useAppState();
    const colors = getThemeColors(themeMode);
    const [selectedExercise, setSelectedExercise] = useState(null);
    const [difficulty, setDifficulty] = useState('beginner');
    const [targetReps, setTargetReps] = useState(10);
    const [sleepHours, setSleepHours] = useState(7.5);
    const [muscleSoreness, setMuscleSoreness] = useState(2);
    const [stress, setStress] = useState(2);
    const [externalLoadKg, setExternalLoadKg] = useState(0);
    const [isCheckingReadiness, setIsCheckingReadiness] = useState(false);
    useEffect(() => {
        loadExercises();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    const resolvedExercises = useMemo(() => {
        if (!availableExercises?.length) {
            return exercises;
        }
        const preferredOrder = exercises.map((entry) => entry.id);
        return availableExercises
            .map((item) => {
                const local = exercises.find((entry) => entry.id === item.id);
                return {
                    id: item.id,
                    label: item.name,
                    icon: local?.icon || 'barbell-outline',
                    image: local?.image || require('@/images/squat.jpg'),
                };
            })
            .sort((a, b) => {
                const aIndex = preferredOrder.indexOf(a.id);
                const bIndex = preferredOrder.indexOf(b.id);
                if (aIndex !== -1 && bIndex !== -1) {
                    return aIndex - bIndex;
                }
                if (aIndex !== -1) {
                    return -1;
                }
                if (bIndex !== -1) {
                    return 1;
                }
                return a.id.localeCompare(b.id);
            });
    }, [availableExercises]);
    const exerciseRows = useMemo(() => ([
        ...chunk(resolvedExercises, 2),
    ]), [resolvedExercises]);
    const begin = async () => {
        if (!selectedExercise)
            return;
        const meta = getExerciseMetadata(selectedExercise);
        const loadKg = supportsExternalLoad(selectedExercise) ? externalLoadKg : 0;
        const started = await startWorkout({
            exerciseName: selectedExercise,
            difficulty,
            targetSets: 1,
            targetReps,
            measurementType: meta.measurementType,
            metricLabel: meta.metricLabel,
            fatiguePrediction: activeFatiguePrediction,
            readinessContext: {
                sleepHours,
                muscleSoreness,
                stress,
                externalLoadKg: loadKg,
                bodyWeightKg: profile?.weightKg,
            },
        });
        if (started) {
            router.push('/(app)/workout-live');
        }
    };
    const activeFatiguePrediction = latestFatiguePrediction?.exerciseId === selectedExercise
        ? latestFatiguePrediction
        : null;
    const lastExerciseSession = useMemo(
        () => findLatestSessionForExercise(history, selectedExercise),
        [history, selectedExercise],
    );
    const suggestedPlan = useMemo(
        () => buildSuggestedRepTarget({
            prediction: activeFatiguePrediction,
            lastSession: lastExerciseSession,
            fallbackReps: targetReps,
            exerciseId: selectedExercise,
        }),
        [activeFatiguePrediction, lastExerciseSession, selectedExercise, targetReps],
    );
    const selectedMeta = getExerciseMetadata(selectedExercise);
    const featureSnapshot = activeFatiguePrediction?.featureSnapshot ?? {};
    const metricName = selectedMeta.measurementType === 'hold' ? 'seconds' : 'reps';
    const targetStep = selectedMeta.measurementType === 'hold' ? 5 : 1;
    const maxTarget = selectedMeta.measurementType === 'hold' ? 600 : 200;
    const refreshReadiness = async (exerciseId = selectedExercise, fallbackTargetOverride = null) => {
        if (!exerciseId) {
            return;
        }
        const meta = getExerciseMetadata(exerciseId);
        const fallbackTarget = meta.measurementType === 'hold' ? 30 : 10;
        const loadKg = supportsExternalLoad(exerciseId) ? externalLoadKg : 0;
        setIsCheckingReadiness(true);
        const prediction = await predictFatigue({
            exerciseId,
            userContext: {
                sleepHours,
                muscleSoreness,
                stress,
                externalLoadKg: loadKg,
                bodyWeightKg: profile?.weightKg,
            },
            recentWindowDays: 14,
        });
        await loadFatigueHistory(exerciseId);
        if (prediction) {
            const lastSession = findLatestSessionForExercise(history, exerciseId);
            const plan = buildSuggestedRepTarget({
                prediction,
                lastSession,
                fallbackReps: fallbackTargetOverride ?? targetReps ?? fallbackTarget,
                exerciseId,
            });
            setTargetReps(plan.targetReps);
        }
        setIsCheckingReadiness(false);
    };
    const selectExercise = (exerciseId) => {
        if (!supportsExternalLoad(exerciseId)) {
            setExternalLoadKg(0);
        }
        const meta = getExerciseMetadata(exerciseId);
        const defaultTarget = meta.measurementType === 'hold' ? 30 : 10;
        setTargetReps(defaultTarget);
        setSelectedExercise(exerciseId);
        refreshReadiness(exerciseId, defaultTarget);
    };
    const exerciseLabel = resolvedExercises.find((e) => e.id === selectedExercise)?.label ?? '';
    const selectedSupportsLoad = supportsExternalLoad(selectedExercise);
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      {!selectedExercise ? (<ScrollView contentContainerStyle={[styles.exerciseRoot, { paddingHorizontal: H_PADDING, paddingBottom: H_PADDING }]} showsVerticalScrollIndicator={false}>
          <Text style={[styles.title, { color: colors.textPrimary }]}>Workout setup</Text>
          <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
            Tap a workout to continue
          </Text>

          <View style={[styles.gridOuter, { gap: GRID_GAP, marginTop: 12 }]}>
            {exerciseRows.map((row, rowIndex) => (<View key={rowIndex} style={[styles.gridRow, { gap: GRID_GAP }]}>
                {row.map((item) => (<Pressable key={item.id} style={({ pressed }) => [
                        styles.exerciseCard,
                        {
                            backgroundColor: colors.surface,
                            borderColor: colors.border,
                            opacity: pressed ? 0.9 : 1,
                        },
                    ]} onPress={() => selectExercise(item.id)}>
                    <View style={[styles.cardImageWrap, { backgroundColor: colors.surfaceAlt }]}>
                      <Image source={item.image} style={styles.cardImage} contentFit="cover"/>
                    </View>
                    <Text style={[styles.cardLabel, { color: colors.textPrimary }]} numberOfLines={2}>
                      {item.label}
                    </Text>
                  </Pressable>))}
              </View>))}
          </View>
          {latestError ? <Text style={[styles.error, { color: colors.danger }]}>{latestError}</Text> : null}
        </ScrollView>) : (<ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} keyboardShouldPersistTaps="handled">
          <Pressable onPress={() => setSelectedExercise(null)} style={styles.backRow}>
            <Ionicons name="chevron-back" size={22} color={colors.brandAlt}/>
            <Text style={[styles.backText, { color: colors.brandAlt }]}>Change workout</Text>
          </Pressable>

          <Text style={[styles.title, { color: colors.textPrimary }]}>Choose intensity</Text>
          <Text style={[styles.subtitle, { color: colors.textSecondary }]}>{exerciseLabel}</Text>

          <View style={[styles.readinessCard, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <View style={styles.readinessHeader}>
              <View>
                <Text style={[styles.panelTitle, { color: colors.textPrimary }]}>Fatigue check</Text>
                <Text style={[styles.panelHint, { color: colors.textSecondary }]}>
                  {activeFatiguePrediction ? fatigueLabel(activeFatiguePrediction.fatigueLevel) : 'No readiness check yet'}
                </Text>
              </View>
              <View style={[styles.scorePill, { borderColor: colors.border, backgroundColor: colors.surfaceAlt }]}>
                <Text style={[styles.scoreValue, { color: colors.brandAlt }]}>
                  {activeFatiguePrediction?.readinessScore ?? '--'}
                </Text>
                <Text style={[styles.scoreLabel, { color: colors.textSecondary }]}>ready</Text>
              </View>
            </View>

            <View style={styles.readinessInputs}>
              <ReadinessStepper
                label="Sleep"
                value={`${sleepHours.toFixed(1)}h`}
                onDecrease={() => setSleepHours((prev) => clamp(prev - 0.5, 4, 10))}
                onIncrease={() => setSleepHours((prev) => clamp(prev + 0.5, 4, 10))}
                colors={colors}
              />
              <ReadinessStepper
                label="Soreness"
                value={`${muscleSoreness}/5`}
                onDecrease={() => setMuscleSoreness((prev) => clamp(prev - 1, 1, 5))}
                onIncrease={() => setMuscleSoreness((prev) => clamp(prev + 1, 1, 5))}
                colors={colors}
              />
              <ReadinessStepper
                label="Stress"
                value={`${stress}/5`}
                onDecrease={() => setStress((prev) => clamp(prev - 1, 1, 5))}
                onIncrease={() => setStress((prev) => clamp(prev + 1, 1, 5))}
                colors={colors}
              />
              {selectedSupportsLoad ? (
                <ReadinessStepper
                  label="Load"
                  value={`${externalLoadKg} kg`}
                  onDecrease={() => setExternalLoadKg((prev) => clamp(prev - 2, 0, 300))}
                  onIncrease={() => setExternalLoadKg((prev) => clamp(prev + 2, 0, 300))}
                  colors={colors}
                />
              ) : null}
            </View>

            {activeFatiguePrediction ? (
              <>
                <Text style={[styles.recommendationText, { color: colors.textSecondary }]}>
                  {activeFatiguePrediction.recommendation}
                </Text>
                <View style={styles.loadRow}>
                  <View style={[styles.loadBox, { backgroundColor: colors.surfaceAlt }]}>
                    <Text style={[styles.loadValue, { color: colors.textPrimary }]}>
                      {featureSnapshot.reps_7d ?? 0}
                    </Text>
                    <Text style={[styles.loadLabel, { color: colors.textSecondary }]}>recent {metricName}</Text>
                  </View>
                  <View style={[styles.loadBox, { backgroundColor: colors.surfaceAlt }]}>
                    <Text style={[styles.loadValue, { color: colors.textPrimary }]}>
                      {featureSnapshot.sessions_7d ?? 0}
                    </Text>
                    <Text style={[styles.loadLabel, { color: colors.textSecondary }]}>recent sessions</Text>
                  </View>
                  <View style={[styles.loadBox, { backgroundColor: colors.surfaceAlt }]}>
                    <Text style={[styles.loadValue, { color: colors.textPrimary }]}>
                      {lastExerciseSession?.reps ?? 0}
                    </Text>
                    <Text style={[styles.loadLabel, { color: colors.textSecondary }]}>last {metricName}</Text>
                  </View>
                  {selectedSupportsLoad ? (
                    <View style={[styles.loadBox, { backgroundColor: colors.surfaceAlt }]}>
                      <Text style={[styles.loadValue, { color: colors.textPrimary }]}>
                        {externalLoadKg}
                      </Text>
                      <Text style={[styles.loadLabel, { color: colors.textSecondary }]}>load kg</Text>
                    </View>
                  ) : null}
                </View>
              </>
            ) : null}

            <View style={[styles.targetPanel, { backgroundColor: colors.surfaceAlt }]}>
              <View style={styles.targetCopy}>
                <Text style={[styles.targetTitle, { color: colors.textPrimary }]}>
                  {suggestedPlan.label}
                </Text>
                <Text style={[styles.targetHint, { color: colors.textSecondary }]}>
                  {suggestedPlan.detail}
                </Text>
              </View>
              <ReadinessStepper
                label={selectedMeta.targetLabel}
                value={`${targetReps} ${suggestedPlan.unit}`}
                onDecrease={() => setTargetReps((prev) => clamp(prev - targetStep, targetStep, maxTarget))}
                onIncrease={() => setTargetReps((prev) => clamp(prev + targetStep, targetStep, maxTarget))}
                colors={colors}
              />
            </View>

            <PrimaryButton
              title={isCheckingReadiness ? 'Checking readiness...' : `Refresh ${selectedMeta.measurementType === 'hold' ? 'hold' : 'fatigue'} check`}
              variant="secondary"
              onPress={() => refreshReadiness()}
              disabled={isCheckingReadiness}
            />
          </View>

          <View style={styles.intensityList}>
            {difficulties.map((item) => {
                const isActive = difficulty === item.id;
                return (<Pressable key={item.id} onPress={() => setDifficulty(item.id)} style={[
                        styles.intensityCard,
                        {
                            backgroundColor: isActive ? colors.surfaceAlt : colors.surface,
                            borderColor: isActive ? colors.brandAlt : colors.border,
                            borderWidth: isActive ? 2 : 1,
                        },
                    ]}>
                  <Text style={[
                        styles.intensityTitle,
                        { color: isActive ? colors.brandAlt : colors.textPrimary },
                    ]}>
                    {item.label}
                  </Text>
                  <Text style={[styles.intensityHint, { color: colors.textSecondary }]}>
                    {item.hint}
                  </Text>
                </Pressable>);
            })}
          </View>

          <PrimaryButton title="Start live session" onPress={begin}/>
          {latestError ? <Text style={[styles.error, { color: colors.danger }]}>{latestError}</Text> : null}
        </ScrollView>)}
    </SafeAreaView>);
}
const styles = StyleSheet.create({
    safe: { flex: 1 },
    exerciseRoot: { flexGrow: 1 },
    scrollContent: { padding: 20, paddingBottom: 32, gap: 16 },
    title: { fontSize: 28, fontWeight: '700', marginTop: 4 },
    subtitle: { fontSize: 15, marginBottom: 4 },
    gridOuter: {},
    gridRow: {
        flexDirection: 'row',
    },
    exerciseCard: {
        flex: 1,
        minWidth: 0,
        minHeight: 172,
        borderRadius: 16,
        borderWidth: 1,
        padding: 10,
        overflow: 'hidden',
        justifyContent: 'space-between',
    },
    cardImageWrap: {
        width: '100%',
        aspectRatio: 1.18,
        borderRadius: 14,
        overflow: 'hidden',
    },
    cardImage: {
        width: '100%',
        height: '100%',
    },
    cardLabel: {
        fontSize: 16,
        fontWeight: '700',
        textAlign: 'center',
        marginTop: 8,
    },
    backRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
        marginBottom: 4,
        alignSelf: 'flex-start',
    },
    backText: { fontSize: 16, fontWeight: '600' },
    readinessCard: {
        borderRadius: 14,
        borderWidth: 1,
        padding: 14,
        gap: 12,
    },
    readinessHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
    },
    panelTitle: { fontSize: 17, fontWeight: '800' },
    panelHint: { fontSize: 13, marginTop: 2 },
    scorePill: {
        width: 76,
        minHeight: 58,
        borderRadius: 8,
        borderWidth: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    scoreValue: { fontSize: 22, fontWeight: '900' },
    scoreLabel: { fontSize: 11, fontWeight: '700', textTransform: 'uppercase' },
    readinessInputs: { gap: 8 },
    stepperRow: {
        minHeight: 34,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 10,
    },
    stepperLabel: { fontSize: 13, fontWeight: '700' },
    stepperControls: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    stepperButton: {
        width: 32,
        height: 32,
        borderRadius: 8,
        borderWidth: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    stepperButtonText: { fontSize: 18, fontWeight: '900', lineHeight: 20 },
    stepperValue: {
        width: 72,
        textAlign: 'center',
        fontSize: 14,
        fontWeight: '800',
    },
    recommendationText: { lineHeight: 19 },
    loadRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
    loadBox: {
        flexBasis: '47%',
        flexGrow: 1,
        minHeight: 58,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 6,
    },
    loadValue: { fontSize: 17, fontWeight: '900' },
    loadLabel: { fontSize: 10, fontWeight: '700', textAlign: 'center', marginTop: 2 },
    targetPanel: {
        borderRadius: 10,
        padding: 12,
        gap: 8,
    },
    targetCopy: { gap: 2 },
    targetTitle: { fontSize: 15, fontWeight: '800' },
    targetHint: { fontSize: 12, lineHeight: 17 },
    intensityList: { gap: 10 },
    intensityCard: {
        borderRadius: 14,
        padding: 16,
        gap: 4,
    },
    intensityTitle: { fontSize: 17, fontWeight: '700' },
    intensityHint: { fontSize: 13 },
    error: { fontSize: 13, marginTop: 6 },
});
