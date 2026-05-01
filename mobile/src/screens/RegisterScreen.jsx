import { Link, useRouter } from 'expo-router';
import React, { useState } from 'react';
import { SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { Input } from '../components/Input';
import { PrimaryButton } from '../components/PrimaryButton';
import { useAppState } from '../state/app-state';
import { colors } from '../theme/colors';
export function RegisterScreen() {
    const router = useRouter();
    const { register, isBusy, latestError, clearError } = useAppState();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const mismatch = Boolean(password && confirmPassword && password !== confirmPassword);
    const handleRegister = async () => {
        clearError();
        if (mismatch)
            return;
        const user = await register({ email, password });
        if (user) {
            router.replace('/(app)/onboarding-profile');
        }
    };
    return (<SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Text style={styles.title}>Create account</Text>
        <Text style={styles.subtitle}>Start tracking your exercise progress</Text>

        <Input label="Email" value={email} placeholder="you@example.com" onChangeText={setEmail}/>
        <Input label="Password" value={password} placeholder="••••••••" secureTextEntry onChangeText={setPassword}/>
        <Input label="Confirm password" value={confirmPassword} placeholder="••••••••" secureTextEntry onChangeText={setConfirmPassword}/>

        {mismatch ? <Text style={styles.error}>Passwords do not match.</Text> : null}
        {latestError ? <Text style={styles.error}>{latestError}</Text> : null}

        <PrimaryButton title={isBusy ? 'Creating account...' : 'Create account'} onPress={handleRegister} disabled={isBusy || mismatch}/>

        <Text style={styles.caption}>
          Already have an account? <Link href="/(auth)/login" style={styles.link}>Sign in</Link>
        </Text>
      </View>
    </SafeAreaView>);
}
const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: colors.background },
    container: { flex: 1, padding: 20, gap: 14, justifyContent: 'center' },
    title: { color: colors.textPrimary, fontSize: 30, fontWeight: '700' },
    subtitle: { color: colors.textSecondary, fontSize: 16, marginBottom: 12 },
    caption: { color: colors.textSecondary, textAlign: 'center', marginTop: 4 },
    link: { color: colors.brandAlt, fontWeight: '600' },
    error: { color: colors.danger, fontSize: 14 },
});
