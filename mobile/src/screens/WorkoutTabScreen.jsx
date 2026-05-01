import { useRouter } from 'expo-router';
import React from 'react';
import { SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { Card } from '@/src/components/Card';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { colors } from '@/src/theme/colors';
export function WorkoutTabScreen() {
    const router = useRouter();
    return (<SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Text style={styles.title}>Workout</Text>
        <Text style={styles.subtitle}>Pick a flow and start training</Text>

        <Card style={styles.card}>
          <Text style={styles.cardTitle}>Quick start</Text>
          <Text style={styles.cardText}>
            Jump straight into setup and begin a guided session with live feedback.
          </Text>
          <PrimaryButton title="Start new workout" onPress={() => router.push('/(app)/workout-setup')}/>
        </Card>

        <PrimaryButton title="View past sessions" variant="secondary" onPress={() => router.push('/(app)/history')}/>
      </View>
    </SafeAreaView>);
}
const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: colors.background },
    container: { flex: 1, padding: 20, gap: 14 },
    title: { color: colors.textPrimary, fontSize: 28, fontWeight: '700', marginTop: 10 },
    subtitle: { color: colors.textSecondary },
    card: { gap: 12 },
    cardTitle: { color: colors.textPrimary, fontWeight: '700', fontSize: 16 },
    cardText: { color: colors.textSecondary, lineHeight: 20 },
});
