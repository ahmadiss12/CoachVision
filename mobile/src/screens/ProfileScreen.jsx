import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import React from 'react';
import { Alert, Pressable, SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { Card } from '@/src/components/Card';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { useAppState } from '@/src/state/app-state';
import { colors } from '@/src/theme/colors';
export function ProfileScreen() {
    const router = useRouter();
    const { authUser, profile, updateProfileAvatar } = useAppState();
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
    return (<SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Text style={styles.title}>Profile</Text>

        <View style={styles.avatarSection}>
          {profile?.avatarUri ? (<Image source={{ uri: profile.avatarUri }} style={styles.avatar} contentFit="cover"/>) : (<View style={[styles.avatar, styles.avatarFallback]}>
              <Ionicons name="person" size={54} color={colors.textSecondary}/>
            </View>)}

          <Text style={styles.userName}>{authUser?.email?.split('@')[0] ?? 'Athlete'}</Text>
          <Text style={styles.userEmail}>{authUser?.email ?? 'No email'}</Text>

          <Pressable style={styles.editAvatarButton} onPress={handleUploadAvatar}>
            <Ionicons name="camera" size={16} color={colors.textPrimary}/>
            <Text style={styles.editAvatarText}>Edit photo</Text>
          </Pressable>
        </View>

        <Card style={styles.infoCard}>
          <InfoRow label="Date of birth" value={profile?.dateOfBirth ?? '--'}/>
          <InfoRow label="Age" value={profile ? `${profile.age}` : '--'}/>
          <InfoRow label="Height" value={profile ? `${profile.heightCm} cm` : '--'}/>
          <InfoRow label="Weight" value={profile ? `${profile.weightKg} kg` : '--'}/>
          <InfoRow label="Body fat" value={profile ? `${profile.bodyFatPercent}%` : '--'}/>
          <InfoRow label="BMI" value={profile ? `${profile.bmi}` : '--'}/>
        </Card>

        <PrimaryButton title="Edit body data" variant="secondary" onPress={() => router.push('/(app)/profile')}/>
        <PrimaryButton title="Track body progress" onPress={() => router.push('/(app)/body-progress')}/>
      </View>
    </SafeAreaView>);
}
function InfoRow({ label, value }) {
    return (<View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Text style={styles.rowValue}>{value}</Text>
    </View>);
}
const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: colors.background },
    container: { flex: 1, padding: 20, gap: 14 },
    title: { color: colors.textPrimary, fontSize: 28, fontWeight: '700', marginTop: 10 },
    avatarSection: { alignItems: 'center', gap: 8, marginTop: 4, marginBottom: 8 },
    avatar: { width: 120, height: 120, borderRadius: 60, borderWidth: 1, borderColor: colors.border },
    avatarFallback: {
        backgroundColor: colors.surface,
        alignItems: 'center',
        justifyContent: 'center',
    },
    userName: { color: colors.textPrimary, fontWeight: '700', fontSize: 20, textTransform: 'capitalize' },
    userEmail: { color: colors.textSecondary, fontSize: 13 },
    editAvatarButton: {
        marginTop: 4,
        flexDirection: 'row',
        gap: 6,
        alignItems: 'center',
        backgroundColor: colors.surfaceAlt,
        borderWidth: 1,
        borderColor: colors.border,
        borderRadius: 999,
        paddingVertical: 8,
        paddingHorizontal: 14,
    },
    editAvatarText: { color: colors.textPrimary, fontWeight: '600' },
    infoCard: { gap: 8 },
    row: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingVertical: 6,
        borderBottomWidth: StyleSheet.hairlineWidth,
        borderBottomColor: colors.border,
    },
    rowLabel: { color: colors.textSecondary },
    rowValue: { color: colors.textPrimary, fontWeight: '600' },
});
