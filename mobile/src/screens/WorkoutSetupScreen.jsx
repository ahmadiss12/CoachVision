import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import React, { useEffect, useMemo, useState } from 'react';
import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View, } from 'react-native';
import { PrimaryButton } from '../components/PrimaryButton';
import { useAppState } from '../state/app-state';
import { getThemeColors } from '../theme/colors';
const exercises = [
    { id: 'squat', label: 'Squat', icon: 'body-outline', image: require('@/images/squat.jpg') },
    { id: 'pushup', label: 'Push-up', icon: 'hand-left-outline', image: require('@/images/pushup.jpg') },
    { id: 'lunge', label: 'Lunge', icon: 'walk-outline', image: require('@/images/lunge.jpg') },
    { id: 'plank', label: 'Plank', icon: 'male-outline', image: require('@/images/8e1aplank.jpg') },
    { id: 'jumping_jack', label: 'Jumping jack', icon: 'flash-outline', image: require('@/images/jumpingjack.jpg') },
    { id: 'high_knees', label: 'High knees', icon: 'trending-up-outline', image: require('@/images/highKnees.jpg') },
];
const difficulties = [
    { id: 'beginner', label: 'Beginner', hint: 'Lighter load, more rest' },
    { id: 'intermediate', label: 'Intermediate', hint: 'Balanced challenge' },
    { id: 'advanced', label: 'Advanced', hint: 'Harder thresholds' },
];
const GRID_GAP = 12;
const H_PADDING = 20;
export function WorkoutSetupScreen() {
    const router = useRouter();
    const { startWorkout, loadExercises, availableExercises, latestError, themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const [selectedExercise, setSelectedExercise] = useState(null);
    const [difficulty, setDifficulty] = useState('beginner');
    useEffect(() => {
        loadExercises();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    const resolvedExercises = useMemo(() => {
        if (!availableExercises?.length) {
            return exercises;
        }
        return availableExercises.map((item) => {
            const local = exercises.find((entry) => entry.id === item.id);
            return {
                id: item.id,
                label: item.name,
                icon: local?.icon || 'barbell-outline',
                image: local?.image || require('@/images/squat.jpg'),
            };
        });
    }, [availableExercises]);
    const exerciseRows = useMemo(() => ([
        resolvedExercises.slice(0, 2),
        resolvedExercises.slice(2, 4),
        resolvedExercises.slice(4, 6),
    ]), [resolvedExercises]);
    const begin = async () => {
        if (!selectedExercise)
            return;
        const started = await startWorkout({ exerciseName: selectedExercise, difficulty });
        if (started) {
            router.push('/(app)/workout-live');
        }
    };
    const exerciseLabel = resolvedExercises.find((e) => e.id === selectedExercise)?.label ?? '';
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      {!selectedExercise ? (<View style={[styles.exerciseRoot, { paddingHorizontal: H_PADDING, paddingBottom: H_PADDING }]}>
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
                    ]} onPress={() => setSelectedExercise(item.id)}>
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
        </View>) : (<ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false} keyboardShouldPersistTaps="handled">
          <Pressable onPress={() => setSelectedExercise(null)} style={styles.backRow}>
            <Ionicons name="chevron-back" size={22} color={colors.brandAlt}/>
            <Text style={[styles.backText, { color: colors.brandAlt }]}>Change workout</Text>
          </Pressable>

          <Text style={[styles.title, { color: colors.textPrimary }]}>Choose intensity</Text>
          <Text style={[styles.subtitle, { color: colors.textSecondary }]}>{exerciseLabel}</Text>

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
    exerciseRoot: { flex: 1 },
    scrollContent: { padding: 20, paddingBottom: 32, gap: 16 },
    title: { fontSize: 28, fontWeight: '700', marginTop: 4 },
    subtitle: { fontSize: 15, marginBottom: 4 },
    gridOuter: {
        flex: 1,
        minHeight: 0,
    },
    gridRow: {
        flex: 1,
        flexDirection: 'row',
        minHeight: 0,
    },
    exerciseCard: {
        flex: 1,
        minWidth: 0,
        minHeight: 0,
        borderRadius: 16,
        borderWidth: 1,
        padding: 10,
        overflow: 'hidden',
        justifyContent: 'space-between',
    },
    cardImageWrap: {
        flex: 1,
        width: '100%',
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
