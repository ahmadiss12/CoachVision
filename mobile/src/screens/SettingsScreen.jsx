import { useRouter } from 'expo-router';
import React from 'react';
import { Pressable, SafeAreaView, StyleSheet, Switch, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { getBmiCategoryLabel } from '@/src/components/BmiGauge';
import { Card } from '@/src/components/Card';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
export function SettingsScreen() {
    const router = useRouter();
    const { authUser, history, logout, profile, themeMode, toggleThemeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const bmiCategory = getBmiCategoryLabel(profile?.bmi ?? null);
    const displayName = authUser?.email?.split('@')[0] ?? 'Athlete';
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <View style={styles.container}>
        <Text style={[styles.title, { color: colors.textPrimary }]}>Settings</Text>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>Account and app preferences</Text>

        <Pressable style={[styles.profileHeader, { backgroundColor: colors.surface, borderColor: colors.border }]} onPress={() => router.push('/(app)/(tabs)/profile')}>
          <View style={[styles.profileIcon, { backgroundColor: colors.surfaceAlt, borderColor: colors.border }]}>
            <Ionicons name="person" size={28} color={colors.textPrimary}/>
          </View>
          <View style={styles.profileInfo}>
            <Text style={[styles.profileName, { color: colors.textPrimary }]}>{displayName}</Text>
            <Text style={[styles.profileEmail, { color: colors.textSecondary }]}>
              {authUser?.email ?? 'Not signed in'}
            </Text>
          </View>
          <Ionicons name="chevron-forward" size={18} color={colors.textSecondary}/>
        </Pressable>

        <Text style={[styles.sectionTitle, { color: colors.textSecondary }]}>Account settings</Text>
        <Card style={styles.listCard}>
          <SettingRow label="Edit profile details" value={profile ? `${profile.heightCm} cm • ${profile.weightKg} kg` : 'Not set'} onPress={() => router.push('/(app)/profile')} textPrimary={colors.textPrimary} textSecondary={colors.textSecondary} borderColor={colors.border}/>
          <SettingRow label="BMI and body composition" value={profile ? `BMI ${profile.bmi} • ${bmiCategory}` : 'Not set'} onPress={() => router.push('/(app)/profile')} textPrimary={colors.textPrimary} textSecondary={colors.textSecondary} borderColor={colors.border}/>
          <SettingRow label="Body progress tracker" value="Update weight and body fat" onPress={() => router.push('/(app)/body-progress')} textPrimary={colors.textPrimary} textSecondary={colors.textSecondary} borderColor={colors.border}/>
          <SettingRow label="Goal settings" value="Target weight, body fat, weekly sessions" onPress={() => router.push('/(app)/goals')} textPrimary={colors.textPrimary} textSecondary={colors.textSecondary} borderColor={colors.border}/>
          <SettingRow label="Workout history" value={`${history.length} sessions`} onPress={() => router.push('/(app)/history')} textPrimary={colors.textPrimary} textSecondary={colors.textSecondary} borderColor={colors.border}/>
        </Card>

        <Card style={styles.preferenceCard}>
          <View style={styles.preferenceRow}>
            <View>
              <Text style={[styles.rowLabel, { color: colors.textPrimary }]}>Light mode</Text>
              <Text style={[styles.rowValue, { color: colors.textSecondary }]}>
                Switch app theme between dark and light
              </Text>
            </View>
            <Switch value={themeMode === 'light'} onValueChange={toggleThemeMode} thumbColor={themeMode === 'light' ? colors.brandAlt : '#f4f3f4'} trackColor={{ false: '#767577', true: '#A5B4FC' }}/>
          </View>
        </Card>

        <PrimaryButton title="Log out" variant="danger" onPress={logout}/>
      </View>
    </SafeAreaView>);
}
function SettingRow({ label, value, onPress, textPrimary, textSecondary, borderColor, }) {
    return (<Pressable onPress={onPress} style={({ pressed }) => [styles.row, { borderBottomColor: borderColor }, pressed && styles.rowPressed]}>
      <View>
        <Text style={[styles.rowLabel, { color: textPrimary }]}>{label}</Text>
        <Text style={[styles.rowValue, { color: textSecondary }]}>{value}</Text>
      </View>
      <Ionicons name="chevron-forward" size={18} color={textSecondary}/>
    </Pressable>);
}
const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { flex: 1, padding: 20, gap: 14 },
    title: { fontSize: 28, fontWeight: '700', marginTop: 10 },
    subtitle: {},
    profileHeader: {
        borderWidth: 1,
        borderRadius: 14,
        padding: 14,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
    },
    profileIcon: {
        width: 50,
        height: 50,
        borderRadius: 25,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
    },
    profileInfo: { flex: 1 },
    profileName: { fontSize: 17, fontWeight: '700', textTransform: 'capitalize' },
    profileEmail: { fontSize: 13, marginTop: 2 },
    sectionTitle: { fontSize: 12, textTransform: 'uppercase', marginTop: 4 },
    listCard: { paddingVertical: 4, paddingHorizontal: 0 },
    preferenceCard: { paddingVertical: 8 },
    preferenceRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
    },
    row: {
        paddingHorizontal: 14,
        paddingVertical: 12,
        borderBottomWidth: StyleSheet.hairlineWidth,
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    rowPressed: { opacity: 0.8 },
    rowLabel: { fontSize: 15, fontWeight: '600' },
    rowValue: { fontSize: 13, marginTop: 2 },
});
