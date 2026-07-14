import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useCallback, useEffect, useState } from 'react';
import { Alert, FlatList, Pressable, RefreshControl, SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { Card } from '@/src/components/Card';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { endClientLink, listClients } from '@/src/services/api/trainer';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

export function TrainerClientsScreen() {
    const router = useRouter();
    const { themeMode, ensureFreshAccessToken } = useAppState();
    const colors = getThemeColors(themeMode);
    const [clients, setClients] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const refresh = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            setClients(await listClients());
        } catch (err) {
            if (err?.status === 401 && (await ensureFreshAccessToken())) {
                try {
                    setClients(await listClients());
                } catch (retryErr) {
                    setError(retryErr?.message || 'Failed to load clients.');
                }
            } else {
                setError(err?.message || 'Failed to load clients.');
            }
        } finally {
            setIsLoading(false);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        refresh();
    }, [refresh]);

    const confirmEndLink = (client) => {
        Alert.alert(
            'End coaching link',
            `Stop coaching ${client.displayName}? You will lose access to their workout history.`,
            [
                { text: 'Cancel', style: 'cancel' },
                {
                    text: 'End link',
                    style: 'destructive',
                    onPress: async () => {
                        try {
                            await endClientLink(client.clientId);
                            refresh();
                        } catch (err) {
                            Alert.alert('Failed', err?.message || 'Could not end the link.');
                        }
                    },
                },
            ],
        );
    };

    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <View style={styles.container}>
        <ScreenHeader
          icon="people-outline"
          title="My clients"
          subtitle={`${clients.length} active ${clients.length === 1 ? 'client' : 'clients'}`}
        />
        {error ? (
          <Card style={styles.emptyCard}>
            <Ionicons name="cloud-offline-outline" size={28} color={colors.danger} />
            <Text style={[styles.empty, { color: colors.danger }]}>{error}</Text>
          </Card>
        ) : null}
        {!error && clients.length === 0 && !isLoading ? (
          <Card style={styles.emptyCard}>
            <View style={[styles.emptyIcon, { backgroundColor: colors.surfaceAlt }]}>
              <Ionicons name="people-outline" size={28} color={colors.textSecondary} />
            </View>
            <Text style={[styles.emptyTitle, { color: colors.textPrimary }]}>No clients yet</Text>
            <Text style={[styles.empty, { color: colors.textSecondary }]}>
              Create an invite code in the Invites tab and share it with your client.
            </Text>
          </Card>
        ) : (
          <FlatList
            data={clients}
            keyExtractor={(item) => item.clientId}
            refreshControl={<RefreshControl refreshing={isLoading} onRefresh={refresh} tintColor={colors.brandAlt} />}
            contentContainerStyle={styles.list}
            showsVerticalScrollIndicator={false}
            renderItem={({ item }) => (
              <Pressable
                onPress={() =>
                  router.push({
                    pathname: '/(app)/client-sessions',
                    params: { clientId: item.clientId, clientName: item.displayName },
                  })
                }
                onLongPress={() => confirmEndLink(item)}
              >
                <Card style={styles.item}>
                  <View style={[styles.itemIcon, { backgroundColor: `${colors.brandAlt}22` }]}>
                    <Ionicons name="person-outline" size={20} color={colors.brandAlt} />
                  </View>
                  <View style={styles.itemCopy}>
                    <Text style={[styles.itemTitle, { color: colors.textPrimary }]}>{item.displayName}</Text>
                    <Text style={[styles.itemSub, { color: colors.textSecondary }]}>{item.email}</Text>
                    <Text style={[styles.itemSub, { color: colors.textSecondary }]}>
                      Linked {new Date(item.linkedAt).toLocaleDateString()}
                    </Text>
                  </View>
                  <Pressable hitSlop={10} onPress={() => confirmEndLink(item)}>
                    <Ionicons name="close-circle-outline" size={22} color={colors.textSecondary} />
                  </Pressable>
                  <Ionicons name="chevron-forward" size={18} color={colors.textSecondary} />
                </Card>
              </Pressable>
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
