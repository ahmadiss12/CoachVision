import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { colors } from '@/src/theme/colors';
const LOADING_MS = 1800;
export function WelcomeScreen() {
    const router = useRouter();
    const [isLoading, setIsLoading] = useState(true);
    useEffect(() => {
        const timer = setTimeout(() => setIsLoading(false), LOADING_MS);
        return () => clearTimeout(timer);
    }, []);
    return (<SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Text style={styles.title}>CoachVision</Text>
        <Text style={styles.subtitle}>AI workout coach for better form and smarter training.</Text>

        {isLoading ? (<View style={styles.loadingBox}>
            <ActivityIndicator size="large" color={colors.brandAlt}/>
            <Text style={styles.loadingText}>Preparing your experience...</Text>
          </View>) : (<View style={styles.actions}>
            <PrimaryButton title="Sign in" onPress={() => router.push('/(auth)/login')}/>
            <PrimaryButton title="Create account" variant="secondary" onPress={() => router.push('/(auth)/register')}/>
          </View>)}
      </View>
    </SafeAreaView>);
}
const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: colors.background },
    container: { flex: 1, justifyContent: 'center', padding: 20, gap: 14 },
    title: { color: colors.textPrimary, fontSize: 36, fontWeight: '800', textAlign: 'center' },
    subtitle: { color: colors.textSecondary, fontSize: 16, textAlign: 'center', marginBottom: 12 },
    loadingBox: { alignItems: 'center', gap: 12, minHeight: 120, justifyContent: 'center' },
    loadingText: { color: colors.textSecondary, fontSize: 14 },
    actions: { gap: 10 },
});
