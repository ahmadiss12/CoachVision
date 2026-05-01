import { Link, useRouter } from 'expo-router';
import React, { useState } from 'react';
import { SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { Input } from '../components/Input';
import { PrimaryButton } from '../components/PrimaryButton';
import { useAppState } from '../state/app-state';
import { colors } from '../theme/colors';
export function LoginScreen() {
    const router = useRouter();
    const { login, isBusy, latestError, clearError } = useAppState();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const handleLogin = async () => {
        clearError();
        const user = await login({ email, password });
        if (user) {
            router.replace('/(app)/onboarding-profile');
        }
    };
    return (<SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Text style={styles.title}>CoachVision</Text>
        <Text style={styles.subtitle}>Sign in to continue your training</Text>

        <Input label="Email" value={email} placeholder="you@example.com" onChangeText={setEmail}/>
        <Input label="Password" value={password} placeholder="••••••••" secureTextEntry onChangeText={setPassword}/>

        {latestError ? <Text style={styles.error}>{latestError}</Text> : null}

        <PrimaryButton title={isBusy ? 'Signing in...' : 'Sign in'} onPress={handleLogin} disabled={isBusy}/>

        <Text style={styles.caption}>
          New here? <Link href="/(auth)/register" style={styles.link}>Create account</Link>
        </Text>
      </View>
    </SafeAreaView>);
}
const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: colors.background },
    container: { flex: 1, padding: 20, gap: 14, justifyContent: 'center' },
    title: { color: colors.textPrimary, fontSize: 32, fontWeight: '700' },
    subtitle: { color: colors.textSecondary, fontSize: 16, marginBottom: 12 },
    caption: { color: colors.textSecondary, textAlign: 'center', marginTop: 4 },
    link: { color: colors.brandAlt, fontWeight: '600' },
    error: { color: colors.danger, fontSize: 14 },
});
