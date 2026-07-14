import { Ionicons } from '@expo/vector-icons';
import React, { useCallback, useEffect, useState } from 'react';
import { Alert, FlatList, Pressable, RefreshControl, SafeAreaView, Share, StyleSheet, Text, View } from 'react-native';
import { Card } from '@/src/components/Card';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { createInvite, listInvites, revokeInvite } from '@/src/services/api/trainer';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

const STATUS_ICONS = {
    pending: 'hourglass-outline',
    accepted: 'checkmark-circle-outline',
    revoked: 'ban-outline',
    expired: 'time-outline',
};

export function TrainerInvitesScreen() {
    const { themeMode, ensureFreshAccessToken } = useAppState();
    const colors = getThemeColors(themeMode);
    const [invites, setInvites] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isCreating, setIsCreating] = useState(false);

    const statusColors = {
        pending: colors.warning ?? colors.brandAlt,
        accepted: colors.success,
        revoked: colors.danger,
        expired: colors.textSecondary,
    };

    const refresh = useCallback(async () => {
        setIsLoading(true);
        try {
            setInvites(await listInvites());
        } catch (err) {
            if (err?.status === 401 && (await ensureFreshAccessToken())) {
                try {
                    setInvites(await listInvites());
                } catch {
                    // surfaced by empty state
                }
            }
        } finally {
            setIsLoading(false);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        refresh();
    }, [refresh]);

    const shareCode = async (token) => {
        try {
            await Share.share({
                message: `Join me on CoachVision! Open the app, go to Profile → My Coach, and enter this invite code: ${token}`,
            });
        } catch {
            // user cancelled share sheet
        }
    };

    const handleCreate = async () => {
        setIsCreating(true);
        try {
            const invite = await createInvite({});
            await refresh();
            shareCode(invite.token);
        } catch (err) {
            Alert.alert('Failed', err?.message || 'Could not create an invite.');
        } finally {
            setIsCreating(false);
        }
    };

    const handleRevoke = (invite) => {
        Alert.alert('Revoke invite', 'This code will stop working immediately.', [
            { text: 'Cancel', style: 'cancel' },
            {
                text: 'Revoke',
                style: 'destructive',
                onPress: async () => {
                    try {
                        await revokeInvite(invite.id);
                        refresh();
                    } catch (err) {
                        Alert.alert('Failed', err?.message || 'Could not revoke the invite.');
                    }
                },
            },
        ]);
    };

    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <View style={styles.container}>
        <ScreenHeader
          icon="mail-open-outline"
          title="Client invites"
          subtitle="Share a code to onboard a client."
        />
        <PrimaryButton
          icon="add-circle-outline"
          title={isCreating ? 'Creating…' : 'New invite code'}
          onPress={handleCreate}
          disabled={isCreating}
        />
        {invites.length === 0 && !isLoading ? (
          <Card style={styles.emptyCard}>
            <View style={[styles.emptyIcon, { backgroundColor: colors.surfaceAlt }]}>
              <Ionicons name="mail-outline" size={28} color={colors.textSecondary} />
            </View>
            <Text style={[styles.emptyTitle, { color: colors.textPrimary }]}>No invites yet</Text>
            <Text style={[styles.empty, { color: colors.textSecondary }]}>
              Create a code and send it to your client — codes work once and expire after 7 days.
            </Text>
          </Card>
        ) : (
          <FlatList
            data={invites}
            keyExtractor={(item) => item.id}
            refreshControl={<RefreshControl refreshing={isLoading} onRefresh={refresh} tintColor={colors.brandAlt} />}
            contentContainerStyle={styles.list}
            showsVerticalScrollIndicator={false}
            renderItem={({ item }) => (
              <Card style={styles.item}>
                <View style={[styles.itemIcon, { backgroundColor: `${statusColors[item.status]}22` }]}>
                  <Ionicons name={STATUS_ICONS[item.status] ?? 'mail-outline'} size={20} color={statusColors[item.status]} />
                </View>
                <View style={styles.itemCopy}>
                  <Text style={[styles.code, { color: colors.textPrimary }]} numberOfLines={1}>{item.token}</Text>
                  <Text style={[styles.itemSub, { color: statusColors[item.status] }]}>
                    {item.status.toUpperCase()}
                    {item.status === 'pending' ? ` · expires ${new Date(item.expiresAt).toLocaleDateString()}` : ''}
                  </Text>
                  {item.email ? (
                    <Text style={[styles.itemSub, { color: colors.textSecondary }]}>for {item.email}</Text>
                  ) : null}
                </View>
                {item.status === 'pending' ? (
                  <>
                    <Pressable hitSlop={10} onPress={() => shareCode(item.token)}>
                      <Ionicons name="share-social-outline" size={21} color={colors.brandAlt} />
                    </Pressable>
                    <Pressable hitSlop={10} onPress={() => handleRevoke(item)}>
                      <Ionicons name="trash-outline" size={20} color={colors.danger} />
                    </Pressable>
                  </>
                ) : null}
              </Card>
            )}
          />
        )}
      </View>
    </SafeAreaView>);
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { flex: 1, padding: 20, gap: 14 },
    list: { gap: 10, paddingBottom: 24 },
    item: { flexDirection: 'row', alignItems: 'center', gap: 12 },
    itemIcon: {
        width: 44,
        height: 44,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    itemCopy: { flex: 1, minWidth: 0 },
    code: { fontWeight: '900', fontSize: 15 },
    itemSub: { fontSize: 11, fontWeight: '800', marginTop: 2 },
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
