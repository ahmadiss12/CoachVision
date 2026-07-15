import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import React from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card } from '@/src/components/Card';
import { MetricTile } from '@/src/components/MetricTile';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

export function ProfileScreen() {
    const router = useRouter();
    const { authUser, userRole, profile, updateProfileAvatar, themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const displayName = authUser?.email?.split('@')[0] ?? 'Athlete';
    const isClient = userRole === 'client';
    const handleUploadAvatar = async () => {
        const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (!permission.granted) {
            Alert.alert('Permission needed', 'Please allow photo access to upload a profile picture.');
            return;
        }
        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ['images'],
            quality: 0.8,
            allowsEditing: true,
            aspect: [1, 1],
        });
        if (!result.canceled && result.assets[0]?.uri) {
            updateProfileAvatar(result.assets[0].uri);
        }
    };
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>
        <ScreenHeader icon="person-circle-outline" title="Profile" subtitle="Body data and account identity." />

        <Card style={styles.identityCard}>
          <Pressable style={styles.avatarButton} onPress={handleUploadAvatar}>
            {profile?.avatarUri ? (<Image source={{ uri: profile.avatarUri }} style={[styles.avatar, { borderColor: colors.border }]} contentFit="cover"/>) : (<View style={[styles.avatar, styles.avatarFallback, { backgroundColor: colors.surfaceAlt, borderColor: colors.border }]}>
                <Ionicons name="person" size={54} color={colors.textSecondary}/>
              </View>)}
            <View style={[styles.cameraBadge, { backgroundColor: colors.brand, borderColor: colors.surface }]}>
              <Ionicons name="camera" size={15} color="#FFFFFF"/>
            </View>
          </Pressable>
          <View style={styles.identityCopy}>
            <Text style={[styles.userName, { color: colors.textPrimary }]}>{displayName}</Text>
            <Text style={[styles.userEmail, { color: colors.textSecondary }]}>{authUser?.email ?? 'No email'}</Text>
            <View style={[styles.roleBadge, { backgroundColor: `${colors.brandAlt}1A`, borderColor: colors.brandAlt }]}>
              <Ionicons
                name={userRole === 'admin' ? 'shield-checkmark-outline' : userRole === 'trainer' ? 'clipboard-outline' : 'fitness-outline'}
                size={13}
                color={colors.brandAlt}
              />
              <Text style={[styles.roleText, { color: colors.brandAlt }]}>{userRole}</Text>
            </View>
          </View>
        </Card>

        {isClient ? (
          <PrimaryButton
            icon="clipboard-outline"
            title="My coach"
            variant="secondary"
            onPress={() => router.push('/(app)/my-coach')}
          />
        ) : null}

        {isClient ? (
          <>
            <View style={styles.metricGrid}>
              <View style={styles.metricRow}>
                <MetricTile icon="resize-outline" label="Height" value={profile ? `${profile.heightCm} cm` : '--'} tone="info" />
                <MetricTile icon="scale-outline" label="Weight" value={profile ? `${profile.weightKg} kg` : '--'} tone="brand" />
              </View>
              <View style={styles.metricRow}>
                <MetricTile icon="water-outline" label="Body fat" value={profile ? `${profile.bodyFatPercent}%` : '--'} tone="warning" />
                <MetricTile icon="speedometer-outline" label="BMI" value={profile ? `${profile.bmi}` : '--'} tone="success" />
              </View>
            </View>

            <Card style={styles.infoCard}>
              <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>Profile details</Text>
              <InfoRow icon="calendar-outline" label="Date of birth" value={profile?.dateOfBirth ?? '--'} colors={colors}/>
              <InfoRow icon="hourglass-outline" label="Age" value={profile ? `${profile.age}` : '--'} colors={colors}/>
            </Card>

            <View style={styles.actions}>
              <PrimaryButton icon="create-outline" title="Edit body data" variant="secondary" onPress={() => router.push('/(app)/profile-edit')} style={styles.actionButton}/>
              <PrimaryButton icon="analytics-outline" title="Track progress" onPress={() => router.push('/(app)/body-progress')} style={styles.actionButton}/>
            </View>
          </>
        ) : null}
      </ScrollView>
    </SafeAreaView>);
}

function InfoRow({ icon, label, value, colors }) {
    return (<View style={[styles.row, { borderBottomColor: colors.border }]}>
      <Ionicons name={icon} size={18} color={colors.brandAlt} />
      <Text style={[styles.rowLabel, { color: colors.textSecondary }]}>{label}</Text>
      <Text style={[styles.rowValue, { color: colors.textPrimary }]}>{value}</Text>
    </View>);
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { padding: 20, gap: 14, paddingBottom: 110 },
    identityCard: { alignItems: 'center', gap: 12 },
    avatarButton: { position: 'relative' },
    avatar: { width: 118, height: 118, borderRadius: 59, borderWidth: 1 },
    avatarFallback: {
        alignItems: 'center',
        justifyContent: 'center',
    },
    cameraBadge: {
        position: 'absolute',
        right: 3,
        bottom: 6,
        width: 34,
        height: 34,
        borderRadius: 17,
        borderWidth: 2,
        alignItems: 'center',
        justifyContent: 'center',
    },
    identityCopy: { alignItems: 'center', gap: 4 },
    roleBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 5,
        borderWidth: 1,
        borderRadius: 999,
        paddingHorizontal: 10,
        paddingVertical: 4,
        marginTop: 4,
    },
    roleText: { fontSize: 12, fontWeight: '900', textTransform: 'uppercase' },
    userName: { fontWeight: '900', fontSize: 22, textTransform: 'capitalize' },
    userEmail: { fontSize: 13, fontWeight: '700' },
    metricGrid: { gap: 10 },
    metricRow: { flexDirection: 'row', gap: 10 },
    infoCard: { gap: 8 },
    sectionTitle: { fontSize: 17, fontWeight: '900', marginBottom: 2 },
    row: {
        minHeight: 44,
        flexDirection: 'row',
        gap: 10,
        alignItems: 'center',
        borderBottomWidth: StyleSheet.hairlineWidth,
    },
    rowLabel: { flex: 1, fontWeight: '700' },
    rowValue: { fontWeight: '900' },
    actions: { flexDirection: 'row', gap: 10 },
    actionButton: { flex: 1 },
});
