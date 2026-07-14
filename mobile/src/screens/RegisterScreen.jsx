import { Ionicons } from '@expo/vector-icons';
import { Link, useRouter } from 'expo-router';
import React, { useState } from 'react';
import { Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';
import { Card } from '../components/Card';
import { Input } from '../components/Input';
import { PrimaryButton } from '../components/PrimaryButton';
import { validateRegisterFields } from '../services/validation/auth-form';
import { useAppState } from '../state/app-state';
import { getThemeColors } from '../theme/colors';

export function RegisterScreen() {
    const router = useRouter();
    const { register, isBusy, latestError, clearError, themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [role, setRole] = useState('client');
    const [formError, setFormError] = useState(null);
    const mismatch = Boolean(password && confirmPassword && password !== confirmPassword);
    const clearFormError = () => {
        setFormError(null);
        clearError();
    };
    const handleRegister = async () => {
        clearFormError();
        const validationError = validateRegisterFields({ email, password, confirmPassword });
        if (validationError) {
            setFormError(validationError);
            return;
        }
        const user = await register({ email: email.trim(), password, role });
        if (user) {
            if (user.role === 'client') {
                router.replace('/(app)/onboarding-profile');
            } else {
                router.replace('/(app)/(tabs)');
            }
        }
    };
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <View style={styles.header}>
          <View style={[styles.iconMark, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <Ionicons name="person-add-outline" size={30} color={colors.brandAlt} />
          </View>
          <Text style={[styles.title, { color: colors.textPrimary }]}>Create account</Text>
          <Text style={[styles.subtitle, { color: colors.textSecondary }]}>Build your personal training profile.</Text>
        </View>

        <Card style={styles.formCard}>
          <View style={styles.roleRow}>
            <RoleOption
              icon="fitness-outline"
              title="Client"
              subtitle="Train with AI feedback"
              selected={role === 'client'}
              colors={colors}
              onPress={() => setRole('client')}
            />
            <RoleOption
              icon="clipboard-outline"
              title="Trainer"
              subtitle="Coach your clients"
              selected={role === 'trainer'}
              colors={colors}
              onPress={() => setRole('trainer')}
            />
          </View>
          <Input
            icon="mail-outline"
            label="Email"
            value={email}
            placeholder="you@example.com"
            keyboardType="email-address"
            textContentType="emailAddress"
            onChangeText={(value) => {
                setEmail(value);
                if (formError || latestError)
                    clearFormError();
            }}
          />
          <Input
            icon="key-outline"
            label="Password"
            value={password}
            placeholder="Password"
            secureTextEntry
            textContentType="newPassword"
            onChangeText={(value) => {
                setPassword(value);
                if (formError || latestError)
                    clearFormError();
            }}
          />
          <Input
            icon="shield-checkmark-outline"
            label="Confirm password"
            value={confirmPassword}
            placeholder="Repeat password"
            secureTextEntry
            textContentType="newPassword"
            onChangeText={(value) => {
                setConfirmPassword(value);
                if (formError || latestError)
                    clearFormError();
            }}
          />

          {formError ? <InlineError message={formError} colors={colors} /> : null}
          {mismatch ? <InlineError message="Passwords do not match." colors={colors} /> : null}
          {latestError ? <InlineError message={latestError} colors={colors} /> : null}

          <PrimaryButton
            icon="arrow-forward-circle-outline"
            iconPosition="right"
            title={isBusy ? 'Creating account...' : 'Create account'}
            onPress={handleRegister}
            disabled={isBusy || mismatch}
          />
        </Card>

        <Text style={[styles.caption, { color: colors.textSecondary }]}>
          Already have an account? <Link href="/(auth)/login" style={[styles.link, { color: colors.brandAlt }]}>Sign in</Link>
        </Text>
      </ScrollView>
    </SafeAreaView>);
}

function RoleOption({ icon, title, subtitle, selected, colors, onPress }) {
    return (
      <Pressable
        onPress={onPress}
        style={[
            styles.roleOption,
            {
                backgroundColor: selected ? `${colors.brandAlt}1A` : colors.background,
                borderColor: selected ? colors.brandAlt : colors.border,
            },
        ]}
      >
        <Ionicons name={icon} size={22} color={selected ? colors.brandAlt : colors.textSecondary} />
        <Text style={[styles.roleTitle, { color: selected ? colors.brandAlt : colors.textPrimary }]}>{title}</Text>
        <Text style={[styles.roleSubtitle, { color: colors.textSecondary }]}>{subtitle}</Text>
      </Pressable>
    );
}

function InlineError({ message, colors }) {
    return (
      <View style={[styles.errorBox, { backgroundColor: `${colors.danger}18`, borderColor: `${colors.danger}55` }]}>
        <Ionicons name="alert-circle-outline" size={18} color={colors.danger} />
        <Text style={[styles.error, { color: colors.danger }]}>{message}</Text>
      </View>
    );
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
    roleRow: { flexDirection: 'row', gap: 10 },
    roleOption: {
        flex: 1,
        borderWidth: 1.5,
        borderRadius: 12,
        padding: 12,
        alignItems: 'center',
        gap: 4,
    },
    roleTitle: { fontSize: 15, fontWeight: '900' },
    roleSubtitle: { fontSize: 11, fontWeight: '600', textAlign: 'center' },
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
});
