import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect, useRouter } from 'expo-router';
import React, { useCallback, useState } from 'react';
import { Alert, FlatList, Pressable, RefreshControl, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card } from '@/src/components/Card';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { deleteProgram, listPrograms } from '@/src/services/api/programs';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

export function TrainerProgramsScreen() {
    const router = useRouter();
    const { themeMode, ensureFreshAccessToken } = useAppState();
    const colors = getThemeColors(themeMode);
    const [programs, setPrograms] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    const refresh = useCallback(async () => {
        setIsLoading(true);
        try {
            setPrograms(await listPrograms());
        } catch (err) {
            if (err?.status === 401 && (await ensureFreshAccessToken())) {
                try {
                    setPrograms(await listPrograms());
                } catch {
                    // shown via empty state
                }
            }
        } finally {
            setIsLoading(false);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useFocusEffect(
        useCallback(() => {
            refresh();
        }, [refresh]),
    );

    const confirmDelete = (program) => {
        Alert.alert(
            `Delete "${program.name}"?`,
            'Clients assigned to it will lose their plan.',
            [
                { text: 'Cancel', style: 'cancel' },
                {
                    text: 'Delete',
                    style: 'destructive',
                    onPress: async () => {
                        try {
                            await deleteProgram(program.id);
                            refresh();
                        } catch (err) {
                            Alert.alert('Failed', err?.message || 'Could not delete the program.');
                        }
                    },
                },
            ],
        );
    };

    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <View style={styles.container}>
        <ScreenHeader
          icon="reader-outline"
          title="Programs"
          subtitle="Build training plans and assign them to clients."
        />
        <PrimaryButton
          icon="add-circle-outline"
          title="New program"
          onPress={() => router.push('/(app)/program-builder')}
        />
        {programs.length === 0 && !isLoading ? (
          <Card style={styles.emptyCard}>
            <View style={[styles.emptyIcon, { backgroundColor: colors.surfaceAlt }]}>
              <Ionicons name="reader-outline" size={28} color={colors.textSecondary} />
            </View>
            <Text style={[styles.emptyTitle, { color: colors.textPrimary }]}>No programs yet</Text>
            <Text style={[styles.empty, { color: colors.textSecondary }]}>
              Create your first training program, then assign it to a client.
            </Text>
          </Card>
        ) : (
          <FlatList
            data={programs}
            keyExtractor={(item) => item.id}
            refreshControl={<RefreshControl refreshing={isLoading} onRefresh={refresh} tintColor={colors.brandAlt} />}
            contentContainerStyle={styles.list}
            showsVerticalScrollIndicator={false}
            renderItem={({ item }) => {
                const exerciseCount = item.days.reduce((sum, day) => sum + day.exercises.length, 0);
                return (
                  <Card style={styles.item}>
                    <View style={styles.itemHeader}>
                      <View style={[styles.itemIcon, { backgroundColor: `${colors.brandAlt}22` }]}>
                        <Ionicons name="reader-outline" size={20} color={colors.brandAlt} />
                      </View>
                      <View style={styles.itemCopy}>
                        <Text style={[styles.itemTitle, { color: colors.textPrimary }]}>{item.name}</Text>
                        <Text style={[styles.itemSub, { color: colors.textSecondary }]}>
                          {item.weeks} week{item.weeks > 1 ? 's' : ''} · {item.days.length} training day
                          {item.days.length === 1 ? '' : 's'} · {exerciseCount} exercise{exerciseCount === 1 ? '' : 's'}
                        </Text>
                      </View>
                      <Pressable hitSlop={8} onPress={() => confirmDelete(item)}>
                        <Ionicons name="trash-outline" size={19} color={colors.danger} />
                      </Pressable>
                    </View>
                    <PrimaryButton
                      icon="person-add-outline"
                      title="Assign to client"
                      variant="secondary"
                      onPress={() =>
                        router.push({
                          pathname: '/(app)/assign-program',
                          params: { programId: item.id, programName: item.name },
                        })
                      }
                    />
                  </Card>
                );
            }}
          />
        )}
      </View>
    </SafeAreaView>);
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { flex: 1, padding: 20, gap: 14 },
    list: { gap: 10, paddingBottom: 24 },
    item: { gap: 12 },
    itemHeader: { flexDirection: 'row', alignItems: 'center', gap: 12 },
    itemIcon: {
        width: 44,
        height: 44,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    itemCopy: { flex: 1, minWidth: 0 },
    itemTitle: { fontWeight: '900', fontSize: 16 },
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
