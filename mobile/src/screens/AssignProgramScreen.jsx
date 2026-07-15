import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useCallback, useEffect, useState } from 'react';
import { Alert, FlatList, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card } from '@/src/components/Card';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { assignProgram } from '@/src/services/api/programs';
import { listClients } from '@/src/services/api/trainer';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

export function AssignProgramScreen() {
    const router = useRouter();
    const { programId, programName } = useLocalSearchParams();
    const { themeMode, ensureFreshAccessToken } = useAppState();
    const colors = getThemeColors(themeMode);
    const [clients, setClients] = useState([]);
    const [assigningId, setAssigningId] = useState(null);

    const refresh = useCallback(async () => {
        try {
            setClients(await listClients());
        } catch (err) {
            if (err?.status === 401 && (await ensureFreshAccessToken())) {
                try {
                    setClients(await listClients());
                } catch {
                    // empty state shown
                }
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        refresh();
    }, [refresh]);

    const assign = async (client) => {
        setAssigningId(client.clientId);
        try {
            await assignProgram(programId, client.clientId);
            Alert.alert(
                'Assigned!',
                `${client.displayName} starts "${programName}" today. Any previous plan was replaced.`,
                [{ text: 'OK', onPress: () => router.back() }],
            );
        } catch (err) {
            Alert.alert('Failed', err?.message || 'Could not assign the program.');
        } finally {
            setAssigningId(null);
        }
    };

    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <View style={styles.container}>
        <ScreenHeader
          showBack
          icon="person-add-outline"
          title="Assign program"
          subtitle={programName ? `Choose who trains "${programName}".` : 'Choose a client.'}
        />
        {clients.length === 0 ? (
          <Card style={styles.emptyCard}>
            <Ionicons name="people-outline" size={28} color={colors.textSecondary} />
            <Text style={[styles.empty, { color: colors.textSecondary }]}>
              No active clients — invite one first from the Invites tab.
            </Text>
          </Card>
        ) : (
          <FlatList
            data={clients}
            keyExtractor={(item) => item.clientId}
            contentContainerStyle={styles.list}
            showsVerticalScrollIndicator={false}
            renderItem={({ item }) => (
              <Pressable onPress={() => assign(item)} disabled={assigningId !== null}>
                <Card style={styles.item}>
                  <View style={[styles.itemIcon, { backgroundColor: `${colors.brandAlt}22` }]}>
                    <Ionicons name="person-outline" size={20} color={colors.brandAlt} />
                  </View>
                  <View style={styles.itemCopy}>
                    <Text style={[styles.itemTitle, { color: colors.textPrimary }]}>{item.displayName}</Text>
                    <Text style={[styles.itemSub, { color: colors.textSecondary }]}>{item.email}</Text>
                  </View>
                  {assigningId === item.clientId ? (
                    <Text style={[styles.assigning, { color: colors.brandAlt }]}>Assigning…</Text>
                  ) : (
                    <Ionicons name="arrow-forward-circle-outline" size={24} color={colors.brandAlt} />
                  )}
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
    itemTitle: { fontWeight: '900', fontSize: 15, textTransform: 'capitalize' },
    itemSub: { fontSize: 12, marginTop: 2 },
    assigning: { fontSize: 12, fontWeight: '900' },
    emptyCard: { alignItems: 'center', gap: 10, paddingVertical: 28 },
    empty: { textAlign: 'center', lineHeight: 20 },
});
