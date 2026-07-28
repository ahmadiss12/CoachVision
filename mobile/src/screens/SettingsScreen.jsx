import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import {
    KeyboardAvoidingView,
    Modal,
    Platform,
    Pressable,
    ScrollView,
    StyleSheet,
    Switch,
    Text,
    View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { getBmiCategoryLabel } from '@/src/components/BmiGauge';
import { Card } from '@/src/components/Card';
import { Input } from '@/src/components/Input';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { TrainerDashboardCard } from '@/src/components/TrainerDashboardCard';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

export function SettingsScreen() {
    const router = useRouter();
    const {
        authUser,
        deleteAccount,
        history,
        isBusy,
        logout,
        profile,
        themeMode,
        toggleThemeMode,
        userRole,
    } = useAppState();
    const canUseWebDashboard = userRole === 'trainer' || userRole === 'admin';
    const colors = getThemeColors(themeMode);
    const bmiCategory = getBmiCategoryLabel(profile?.bmi ?? null);
    const displayName = authUser?.email?.split('@')[0] ?? 'Athlete';
    const [isDeleteOpen, setIsDeleteOpen] = useState(false);
    const [deletePassword, setDeletePassword] = useState('');
    const [deleteError, setDeleteError] = useState(null);

    const openDeleteSheet = () => {
        setDeletePassword('');
        setDeleteError(null);
        setIsDeleteOpen(true);
    };

    const closeDeleteSheet = () => {
        if (isBusy) {
            return;
        }
        setIsDeleteOpen(false);
        setDeletePassword('');
        setDeleteError(null);
    };

    const confirmDelete = async () => {
        setDeleteError(null);
        const result = await deleteAccount(deletePassword);
        if (result?.ok) {
            // Clearing auth state makes AppLayout redirect to the auth stack,
            // so unmounting this modal first avoids a flash of the sheet.
            setIsDeleteOpen(false);
            setDeletePassword('');
            return;
        }
        setDeleteError(result?.message || 'Could not delete the account.');
    };
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>
        <ScreenHeader icon="options-outline" title="Settings" subtitle="Account, profile, and app preferences." />

        <Pressable style={({ pressed }) => [styles.profileHeader, { backgroundColor: colors.surface, borderColor: colors.border, opacity: pressed ? 0.86 : 1 }]} onPress={() => router.push('/(app)/(tabs)/profile')}>
          <View style={[styles.profileIcon, { backgroundColor: colors.surfaceAlt, borderColor: colors.border }]}>
            <Ionicons name="person" size={28} color={colors.brandAlt}/>
          </View>
          <View style={styles.profileInfo}>
            <Text style={[styles.profileName, { color: colors.textPrimary }]}>{displayName}</Text>
            <Text style={[styles.profileEmail, { color: colors.textSecondary }]}>
              {authUser?.email ?? 'Not signed in'}
            </Text>
          </View>
          <Ionicons name="chevron-forward" size={18} color={colors.textSecondary}/>
        </Pressable>

        <Card style={styles.listCard}>
          <SettingRow icon="create-outline" label="Edit body data" value={profile ? `${profile.heightCm} cm | ${profile.weightKg} kg` : 'Not set'} onPress={() => router.push('/(app)/profile-edit')} colors={colors}/>
          <SettingRow icon="speedometer-outline" label="BMI and body composition" value={profile ? `BMI ${profile.bmi} | ${bmiCategory}` : 'Not set'} onPress={() => router.push('/(app)/profile-edit')} colors={colors}/>
          <SettingRow icon="analytics-outline" label="Body progress tracker" value="Weight and body fat logs" onPress={() => router.push('/(app)/body-progress')} colors={colors}/>
          <SettingRow icon="flag-outline" label="Goal settings" value="Weight, body fat, weekly sessions" onPress={() => router.push('/(app)/goals')} colors={colors}/>
          <SettingRow icon="time-outline" label="Workout history" value={`${history.length} sessions`} onPress={() => router.push('/(app)/history')} colors={colors} isLast/>
        </Card>

        <Card style={styles.preferenceCard}>
          <View style={styles.preferenceRow}>
            <View style={styles.preferenceIconText}>
              <View style={[styles.preferenceIcon, { backgroundColor: colors.surfaceAlt }]}>
                <Ionicons name={themeMode === 'light' ? 'sunny-outline' : 'moon-outline'} size={20} color={colors.brandAlt} />
              </View>
              <View style={styles.preferenceCopy}>
                <Text style={[styles.rowLabel, { color: colors.textPrimary }]}>Light mode</Text>
                <Text style={[styles.rowValue, { color: colors.textSecondary }]}>Switch app theme</Text>
              </View>
            </View>
            <Switch value={themeMode === 'light'} onValueChange={toggleThemeMode} thumbColor={themeMode === 'light' ? colors.brandAlt : '#f4f3f4'} trackColor={{ false: '#6B7280', true: '#99F6E4' }}/>
          </View>
        </Card>

        {canUseWebDashboard ? <TrainerDashboardCard /> : null}

        <PrimaryButton icon="log-out-outline" title="Log out" variant="danger" onPress={logout}/>

        <Card style={[styles.dangerCard, { borderColor: colors.danger }]}>
          <View style={styles.dangerHeader}>
            <Ionicons name="warning-outline" size={18} color={colors.danger} />
            <Text style={[styles.dangerTitle, { color: colors.danger }]}>Danger zone</Text>
          </View>
          <Text style={[styles.dangerCopy, { color: colors.textSecondary }]}>
            Deleting your account permanently removes your profile, workout history,
            body measurements, and coaching links. This cannot be undone.
          </Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Delete my account"
            onPress={openDeleteSheet}
            style={({ pressed }) => [
                styles.deleteButton,
                { borderColor: colors.danger, opacity: pressed ? 0.82 : 1 },
            ]}
          >
            <Ionicons name="trash-outline" size={18} color={colors.danger} />
            <Text style={[styles.deleteButtonText, { color: colors.danger }]}>
              Delete my account
            </Text>
          </Pressable>
        </Card>
      </ScrollView>

      <Modal
        visible={isDeleteOpen}
        transparent
        animationType="fade"
        onRequestClose={closeDeleteSheet}
      >
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          style={styles.modalBackdrop}
        >
          <View style={[styles.modalCard, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <View style={styles.dangerHeader}>
              <Ionicons name="alert-circle-outline" size={22} color={colors.danger} />
              <Text style={[styles.modalTitle, { color: colors.textPrimary }]}>
                Delete account?
              </Text>
            </View>

            <Text style={[styles.dangerCopy, { color: colors.textSecondary }]}>
              This permanently deletes {authUser?.email ?? 'your account'} and all
              associated data. It cannot be undone.
            </Text>

            <Input
              label="Confirm your password"
              value={deletePassword}
              onChangeText={(text) => {
                  setDeletePassword(text);
                  if (deleteError) {
                      setDeleteError(null);
                  }
              }}
              placeholder="Current password"
              secureTextEntry
              icon="lock-closed-outline"
              editable={!isBusy}
              textContentType="password"
            />

            {deleteError ? (
              <Text style={[styles.modalError, { color: colors.danger }]}>{deleteError}</Text>
            ) : null}

            <View style={styles.modalActions}>
              <PrimaryButton
                title="Cancel"
                variant="secondary"
                onPress={closeDeleteSheet}
                disabled={isBusy}
                style={styles.modalAction}
              />
              <PrimaryButton
                title={isBusy ? 'Deleting...' : 'Delete forever'}
                variant="danger"
                icon="trash-outline"
                onPress={confirmDelete}
                disabled={isBusy || deletePassword.length === 0}
                style={styles.modalAction}
              />
            </View>
          </View>
        </KeyboardAvoidingView>
      </Modal>
    </SafeAreaView>);
}

function SettingRow({ icon, label, value, onPress, colors, isLast = false }) {
    return (<Pressable onPress={onPress} style={({ pressed }) => [styles.row, { borderBottomColor: isLast ? 'transparent' : colors.border }, pressed && styles.rowPressed]}>
      <View style={[styles.rowIcon, { backgroundColor: colors.surfaceAlt }]}>
        <Ionicons name={icon} size={18} color={colors.brandAlt}/>
      </View>
      <View style={styles.rowCopy}>
        <Text style={[styles.rowLabel, { color: colors.textPrimary }]}>{label}</Text>
        <Text style={[styles.rowValue, { color: colors.textSecondary }]}>{value}</Text>
      </View>
      <Ionicons name="chevron-forward" size={18} color={colors.textSecondary}/>
    </Pressable>);
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { padding: 20, gap: 14, paddingBottom: 110 },
    profileHeader: {
        borderWidth: 1,
        borderRadius: 8,
        padding: 14,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
    },
    profileIcon: {
        width: 52,
        height: 52,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
    },
    profileInfo: { flex: 1, minWidth: 0 },
    profileName: { fontSize: 17, fontWeight: '900', textTransform: 'capitalize' },
    profileEmail: { fontSize: 13, marginTop: 2, fontWeight: '700' },
    listCard: { paddingVertical: 0, paddingHorizontal: 0, overflow: 'hidden' },
    preferenceCard: { paddingVertical: 12 },
    preferenceRow: {
        minHeight: 52,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
    },
    preferenceIconText: { flex: 1, flexDirection: 'row', alignItems: 'center', gap: 12 },
    preferenceIcon: {
        width: 42,
        height: 42,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    preferenceCopy: { flex: 1, minWidth: 0 },
    row: {
        paddingHorizontal: 14,
        paddingVertical: 12,
        borderBottomWidth: StyleSheet.hairlineWidth,
        flexDirection: 'row',
        gap: 12,
        alignItems: 'center',
    },
    rowPressed: { opacity: 0.8 },
    rowIcon: {
        width: 38,
        height: 38,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    rowCopy: { flex: 1, minWidth: 0 },
    rowLabel: { fontSize: 15, fontWeight: '900' },
    rowValue: { fontSize: 13, marginTop: 2, fontWeight: '700' },
    dangerCard: { gap: 10, borderWidth: 1 },
    dangerHeader: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    dangerTitle: { fontSize: 15, fontWeight: '900' },
    dangerCopy: { fontSize: 13, lineHeight: 19, fontWeight: '600' },
    deleteButton: {
        minHeight: 48,
        borderRadius: 8,
        borderWidth: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
    },
    deleteButtonText: { fontSize: 15, fontWeight: '800' },
    modalBackdrop: {
        flex: 1,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
    },
    modalCard: {
        width: '100%',
        maxWidth: 420,
        borderRadius: 12,
        borderWidth: 1,
        padding: 18,
        gap: 14,
    },
    modalTitle: { fontSize: 18, fontWeight: '900' },
    modalError: { fontSize: 13, fontWeight: '700' },
    modalActions: { flexDirection: 'row', gap: 10 },
    modalAction: { flex: 1 },
});
