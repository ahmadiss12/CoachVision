import { Ionicons } from '@expo/vector-icons';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Alert, FlatList, Pressable, RefreshControl, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card } from '@/src/components/Card';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { changeUserRole, deleteUser, getPlatformStats, listAllUsers } from '@/src/services/api/admin';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

const ROLE_ICONS = {
    client: 'fitness-outline',
    trainer: 'clipboard-outline',
    admin: 'shield-checkmark-outline',
};

export function AdminUsersScreen() {
    const { themeMode, authUser, ensureFreshAccessToken } = useAppState();
    const colors = getThemeColors(themeMode);
    const [users, setUsers] = useState([]);
    const [stats, setStats] = useState(null);
    const [query, setQuery] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const roleColors = {
        client: colors.brandAlt,
        trainer: colors.success,
        admin: colors.warning ?? colors.danger,
    };

    const withAuthRetry = useCallback(async (call) => {
        try {
            return await call();
        } catch (err) {
            if (err?.status === 401 && (await ensureFreshAccessToken())) {
                return call();
            }
            throw err;
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const refresh = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const [userRows, statRow] = await Promise.all([
                withAuthRetry(() => listAllUsers()),
                withAuthRetry(() => getPlatformStats()).catch(() => null),
            ]);
            setUsers(userRows);
            setStats(statRow);
        } catch (err) {
            setError(err?.message || 'Failed to load users.');
        } finally {
            setIsLoading(false);
        }
    }, [withAuthRetry]);

    useEffect(() => {
        refresh();
    }, [refresh]);

    const visibleUsers = useMemo(() => {
        const needle = query.trim().toLowerCase();
        if (!needle) {
            return users;
        }
        return users.filter(
            (user) =>
                user.email.toLowerCase().includes(needle) ||
                user.displayName.toLowerCase().includes(needle) ||
                user.role.includes(needle),
        );
    }, [users, query]);

    const applyRole = async (user, role) => {
        try {
            const updated = await withAuthRetry(() => changeUserRole(user.id, role));
            // Update the row in place from the server response — no full reload needed.
            setUsers((prev) => prev.map((u) => (u.id === updated.id ? updated : u)));
            withAuthRetry(() => getPlatformStats()).then(setStats).catch(() => undefined);
        } catch (err) {
            Alert.alert(
                'Role change failed',
                `${err?.message || 'Unknown error'}${err?.status ? ` (HTTP ${err.status})` : ''}`,
            );
        }
    };

    const pickRole = (user) => {
        if (user.id === authUser?.id) {
            Alert.alert('Not allowed', 'You cannot change your own admin role.');
            return;
        }
        Alert.alert(
            `Change role for ${user.displayName}`,
            `Current role: ${user.role}. A demoted trainer loses access to all their clients.`,
            [
                { text: 'Cancel', style: 'cancel' },
                ...['client', 'trainer', 'admin']
                    .filter((role) => role !== user.role)
                    .map((role) => ({
                        text: `Make ${role}`,
                        onPress: () => applyRole(user, role),
                    })),
            ],
        );
    };

    const confirmDelete = (user) => {
        if (user.id === authUser?.id) {
            Alert.alert('Not allowed', 'You cannot delete your own account.');
            return;
        }
        Alert.alert(
            `Delete ${user.displayName}?`,
            'This permanently removes the account and ALL its data — workouts, metrics, coaching links. This cannot be undone.',
            [
                { text: 'Cancel', style: 'cancel' },
                {
                    text: 'Delete forever',
                    style: 'destructive',
                    onPress: async () => {
                        try {
                            await withAuthRetry(() => deleteUser(user.id));
                            setUsers((prev) => prev.filter((u) => u.id !== user.id));
                            withAuthRetry(() => getPlatformStats()).then(setStats).catch(() => undefined);
                        } catch (err) {
                            Alert.alert('Delete failed', err?.message || 'Unknown error');
                        }
                    },
                },
            ],
        );
    };

    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <View style={styles.container}>
        <ScreenHeader
          icon="shield-checkmark-outline"
          title="User management"
          subtitle={`${users.length} registered ${users.length === 1 ? 'user' : 'users'}`}
        />

        {stats ? (
          <View style={styles.statsRow}>
            <StatPill icon="people-outline" label="Clients" value={stats.clients} colors={colors} tint={colors.brandAlt} />
            <StatPill icon="clipboard-outline" label="Trainers" value={stats.trainers} colors={colors} tint={colors.success} />
            <StatPill icon="barbell-outline" label="Workouts" value={stats.completedSessions} colors={colors} tint={colors.warning ?? colors.danger} />
            <StatPill icon="flash-outline" label="Last 7d" value={stats.sessionsLast7d} colors={colors} tint={colors.brandAlt} />
          </View>
        ) : null}

        <View style={[styles.searchBox, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          <Ionicons name="search-outline" size={18} color={colors.textSecondary} />
          <TextInput
            style={[styles.searchInput, { color: colors.textPrimary }]}
            value={query}
            onChangeText={setQuery}
            placeholder="Search by name, email, or role"
            placeholderTextColor={colors.textSecondary}
            autoCapitalize="none"
            autoCorrect={false}
          />
          {query ? (
            <Pressable hitSlop={8} onPress={() => setQuery('')}>
              <Ionicons name="close-circle" size={18} color={colors.textSecondary} />
            </Pressable>
          ) : null}
        </View>

        {error ? (
          <Card style={styles.emptyCard}>
            <Ionicons name="cloud-offline-outline" size={28} color={colors.danger} />
            <Text style={[styles.empty, { color: colors.danger }]}>{error}</Text>
          </Card>
        ) : (
          <FlatList
            data={visibleUsers}
            keyExtractor={(item) => item.id}
            refreshControl={<RefreshControl refreshing={isLoading} onRefresh={refresh} tintColor={colors.brandAlt} />}
            contentContainerStyle={styles.list}
            showsVerticalScrollIndicator={false}
            renderItem={({ item }) => (
              <Card style={styles.item}>
                <View style={[styles.itemIcon, { backgroundColor: `${roleColors[item.role]}22` }]}>
                  <Ionicons name={ROLE_ICONS[item.role] ?? 'person-outline'} size={20} color={roleColors[item.role]} />
                </View>
                <View style={styles.itemCopy}>
                  <Text style={[styles.itemTitle, { color: colors.textPrimary }]}>
                    {item.displayName}
                    {item.id === authUser?.id ? '  (you)' : ''}
                  </Text>
                  <Text style={[styles.itemSub, { color: colors.textSecondary }]}>{item.email}</Text>
                </View>
                <Pressable
                  onPress={() => pickRole(item)}
                  style={[styles.roleBadge, { backgroundColor: `${roleColors[item.role]}1A`, borderColor: roleColors[item.role] }]}
                >
                  <Text style={[styles.roleText, { color: roleColors[item.role] }]}>{item.role}</Text>
                  <Ionicons name="chevron-down" size={12} color={roleColors[item.role]} />
                </Pressable>
                {item.id !== authUser?.id ? (
                  <Pressable hitSlop={8} onPress={() => confirmDelete(item)}>
                    <Ionicons name="trash-outline" size={19} color={colors.danger} />
                  </Pressable>
                ) : null}
              </Card>
            )}
          />
        )}
      </View>
    </SafeAreaView>);
}

function StatPill({ icon, label, value, colors, tint }) {
    return (
      <View style={[styles.statPill, { backgroundColor: colors.surface, borderColor: colors.border }]}>
        <Ionicons name={icon} size={15} color={tint} />
        <Text style={[styles.statValue, { color: colors.textPrimary }]}>{value}</Text>
        <Text style={[styles.statLabel, { color: colors.textSecondary }]}>{label}</Text>
      </View>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { flex: 1, padding: 20, gap: 14 },
    statsRow: { flexDirection: 'row', gap: 8 },
    statPill: {
        flex: 1,
        borderWidth: 1,
        borderRadius: 10,
        paddingVertical: 8,
        alignItems: 'center',
        gap: 2,
    },
    statValue: { fontSize: 16, fontWeight: '900' },
    statLabel: { fontSize: 10, fontWeight: '800' },
    searchBox: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
        borderWidth: 1,
        borderRadius: 10,
        paddingHorizontal: 12,
        height: 44,
    },
    searchInput: { flex: 1, fontSize: 14, fontWeight: '600' },
    list: { gap: 10, paddingBottom: 24 },
    item: { flexDirection: 'row', alignItems: 'center', gap: 10 },
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
    roleBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
        borderWidth: 1,
        borderRadius: 8,
        paddingHorizontal: 10,
        paddingVertical: 6,
    },
    roleText: { fontSize: 12, fontWeight: '900', textTransform: 'uppercase' },
    emptyCard: { alignItems: 'center', gap: 10, paddingVertical: 28 },
    empty: { textAlign: 'center', lineHeight: 20 },
});
