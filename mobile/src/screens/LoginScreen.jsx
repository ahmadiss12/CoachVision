import { Ionicons } from '@expo/vector-icons';
import { Link, useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Platform, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Card } from '../components/Card';
import { Input } from '../components/Input';
import { PrimaryButton } from '../components/PrimaryButton';
import { useAppState } from '../state/app-state';
import { getThemeColors } from '../theme/colors';

function hasSavedBodyProfile(user) {
    return Boolean(user?.date_of_birth && user?.height_cm && user?.weight_kg);
}

export function LoginScreen() {
    const router = useRouter();
    const { login, loadCurrentUser, saveGoals, isBusy, latestError, clearError, themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const handleLogin = async () => {
        clearError();
        const user = await login({ email, password });
        if (user) {
            router.replace(hasSavedBodyProfile(user) ? '/' : '/(app)/onboarding-profile');
        }
    };
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <View style={styles.header}>
          <View style={[styles.iconMark, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <Ionicons name="lock-closed-outline" size={30} color={colors.brandAlt} />
          </View>
          <Text style={[styles.title, { color: colors.textPrimary }]}>Welcome back</Text>
          <Text style={[styles.subtitle, { color: colors.textSecondary }]}>Sign in to continue training.</Text>
        </View>

        <Card style={styles.formCard}>
          <Input icon="mail-outline" label="Email" value={email} placeholder="you@example.com" onChangeText={setEmail}/>
          <Input icon="key-outline" label="Password" value={password} placeholder="Password" secureTextEntry onChangeText={setPassword}/>

          {latestError ? (
            <View style={[styles.errorBox, { backgroundColor: `${colors.danger}18`, borderColor: `${colors.danger}55` }]}>
              <Ionicons name="alert-circle-outline" size={18} color={colors.danger} />
              <Text style={[styles.error, { color: colors.danger }]}>{latestError}</Text>
            </View>
          ) : null}

          <PrimaryButton
            icon="log-in-outline"
            title={isBusy ? 'Signing in...' : 'Sign in'}
            onPress={handleLogin}
            disabled={isBusy}
            accessibilityLabel="Sign in"
          />

          {__DEV__ && Platform.OS === 'web' ? (
            <Pressable
              style={({ pressed }) => [
                styles.devButton,
                { borderColor: colors.border, backgroundColor: colors.surfaceAlt, opacity: pressed ? 0.82 : 1 },
              ]}
              disabled={isBusy}
              onPress={async () => {
                clearError();
                const user = await login({ email: 'admin@coachvision.test', password: 'Admin1234' });
                if (!user) {
                  return;
                }
                await loadCurrentUser();
                saveGoals({ targetWeightKg: 75, targetBodyFatPercent: 15, weeklyWorkoutTarget: 4 });
                router.replace('/(app)/(tabs)');
              }}
            >
              <Ionicons name="construct-outline" size={16} color={colors.textSecondary} />
              <Text style={[styles.devButtonText, { color: colors.textSecondary }]}>Dev sign in</Text>
            </Pressable>
          ) : null}
        </Card>

        <Text style={[styles.caption, { color: colors.textSecondary }]}>
          New here? <Link href="/(auth)/register" style={[styles.link, { color: colors.brandAlt }]}>Create account</Link>
        </Text>
      </ScrollView>
    </SafeAreaView>);
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { flexGrow: 1, padding: 20, gap: 18, justifyContent: 'center' },
    header: { gap: 8 },
    iconMark: {
        width: 62,
        height: 62,
        borderRadius: 16,
        borderWidth: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    title: { fontSize: 33, lineHeight: 38, fontWeight: '900' },
    subtitle: { fontSize: 16 },
    formCard: { gap: 14 },
    caption: { textAlign: 'center', marginTop: 2, fontWeight: '700' },
    link: { fontWeight: '900' },
    errorBox: {
        borderWidth: 1,
        borderRadius: 8,
        padding: 10,
        flexDirection: 'row',
        gap: 8,
        alignItems: 'center',
    },
    error: { flex: 1, fontSize: 13, fontWeight: '700', lineHeight: 18 },
    devButton: {
        minHeight: 44,
        paddingVertical: 10,
        borderRadius: 8,
        borderWidth: 1,
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'row',
        gap: 8,
    },
    devButtonText: { fontSize: 13, fontWeight: '800' },
});
