import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useCallback, useEffect, useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card } from '@/src/components/Card';
import { Input } from '@/src/components/Input';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { acceptInvite, listMyTrainers } from '@/src/services/api/trainer';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

export function MyCoachScreen() {
    const router = useRouter();
    const { themeMode, ensureFreshAccessToken } = useAppState();
    const colors = getThemeColors(themeMode);
    const [trainers, setTrainers] = useState([]);
    const [code, setCode] = useState('');
    const [isJoining, setIsJoining] = useState(false);

    const refresh = useCallback(async () => {
        try {
            setTrainers(await listMyTrainers());
        } catch (err) {
            if (err?.status === 401 && (await ensureFreshAccessToken())) {
                try {
                    setTrainers(await listMyTrainers());
                } catch {
                    // shown via empty state
                }
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        refresh();
    }, [refresh]);

    const handleJoin = async () => {
        const token = code.trim();
        if (!token) {
            Alert.alert('Missing code', 'Paste the invite code your coach sent you.');
            return;
        }
        setIsJoining(true);
        try {
            const result = await acceptInvite(token);
            setCode('');
            await refresh();
            Alert.alert('Connected!', `${result.trainerName} is now your coach.`);
        } catch (err) {
            Alert.alert('Could not join', err?.message || 'Invalid or expired invite code.');
        } finally {
            setIsJoining(false);
        }
    };

    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <ScreenHeader
          showBack
          icon="clipboard-outline"
          title="My coach"
          subtitle="Connect with a trainer to get guided workouts."
        />

        <Card style={styles.joinCard}>
          <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>Have an invite code?</Text>
          <Input
            icon="key-outline"
            label="Invite code"
            value={code}
            placeholder="Paste the code from your coach"
            autoCapitalize="none"
            autoCorrect={false}
            onChangeText={setCode}
          />
          <PrimaryButton
            icon="link-outline"
            title={isJoining ? 'Connecting…' : 'Connect with coach'}
            onPress={handleJoin}
            disabled={isJoining}
          />
        </Card>

        <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>Current coach</Text>
        {trainers.length === 0 ? (
          <Card style={styles.emptyCard}>
            <View style={[styles.emptyIcon, { backgroundColor: colors.surfaceAlt }]}>
              <Ionicons name="clipboard-outline" size={28} color={colors.textSecondary} />
            </View>
            <Text style={[styles.emptyTitle, { color: colors.textPrimary }]}>No coach yet</Text>
            <Text style={[styles.empty, { color: colors.textSecondary }]}>
              Ask your trainer for an invite code and enter it above.
            </Text>
          </Card>
        ) : (
          trainers.map((trainer) => (
            <Card key={trainer.clientId} style={styles.trainerItem}>
              <View style={[styles.itemIcon, { backgroundColor: `${colors.success}22` }]}>
                <Ionicons name="person-outline" size={20} color={colors.success} />
              </View>
              <View style={styles.itemCopy}>
                <Text style={[styles.itemTitle, { color: colors.textPrimary }]}>{trainer.displayName}</Text>
                <Text style={[styles.itemSub, { color: colors.textSecondary }]}>{trainer.email}</Text>
                <Text style={[styles.itemSub, { color: colors.textSecondary }]}>
                  Coaching you since {new Date(trainer.linkedAt).toLocaleDateString()}
                </Text>
              </View>
              <Pressable
                hitSlop={10}
                onPress={() =>
                  router.push({
                    pathname: '/(app)/chat',
                    params: { partnerId: trainer.clientId, partnerName: trainer.displayName },
                  })
                }
                style={[styles.chatButton, { backgroundColor: `${colors.brandAlt}1A`, borderColor: colors.brandAlt }]}
              >
                <Ionicons name="chatbubbles-outline" size={19} color={colors.brandAlt} />
              </Pressable>
            </Card>
          ))
        )}
      </ScrollView>
    </SafeAreaView>);
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { padding: 20, gap: 14, paddingBottom: 40 },
    joinCard: { gap: 14 },
    sectionTitle: { fontSize: 17, fontWeight: '900' },
    trainerItem: { flexDirection: 'row', alignItems: 'center', gap: 12 },
    chatButton: {
        width: 38,
        height: 38,
        borderRadius: 10,
        borderWidth: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    itemIcon: {
        width: 44,
        height: 44,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    itemCopy: { flex: 1, minWidth: 0 },
    itemTitle: { fontWeight: '900', fontSize: 16, textTransform: 'capitalize' },
    itemSub: { fontSize: 12, marginTop: 2 },
    emptyCard: { alignItems: 'center', gap: 10, paddingVertical: 28 },
    emptyIcon: {
        width: 58,
        height: 58,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    emptyTitle: { fontSize: 18, fontWeight: '900' },
    empty: { textAlign: 'center', lineHeight: 20 },
});
