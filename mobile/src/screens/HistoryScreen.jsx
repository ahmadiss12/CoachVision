import React, { useEffect } from 'react';
import { FlatList, SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { Card } from '@/src/components/Card';
import { useAppState } from '@/src/state/app-state';
import { colors } from '@/src/theme/colors';
function HistoryItem({ item }) {
    return (<Card style={styles.item}>
      <Text style={styles.itemTitle}>
        {item.exerciseName} • {item.difficulty}
      </Text>
      <Text style={styles.itemMeta}>
        Reps {item.reps} • {item.durationSec}s • Score {item.score}
      </Text>
      <Text style={styles.itemDate}>{new Date(item.endedAt).toLocaleString()}</Text>
    </Card>);
}
export function HistoryScreen() {
    const { history, loadHistory } = useAppState();
    useEffect(() => {
        loadHistory();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    return (<SafeAreaView style={styles.safe}>
      <View style={styles.container}>
        <Text style={styles.title}>Workout history</Text>
        {history.length === 0 ? (<Card>
            <Text style={styles.empty}>No sessions yet. Complete a workout to see history.</Text>
          </Card>) : (<FlatList data={history} keyExtractor={(item) => item.id} renderItem={({ item }) => <HistoryItem item={item}/>} contentContainerStyle={styles.list}/>)}
      </View>
    </SafeAreaView>);
}
const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: colors.background },
    container: { flex: 1, padding: 20, gap: 12 },
    title: { color: colors.textPrimary, fontSize: 28, fontWeight: '700', marginTop: 10 },
    empty: { color: colors.textSecondary },
    list: { gap: 10, paddingBottom: 16 },
    item: { gap: 4 },
    itemTitle: { color: colors.textPrimary, fontWeight: '700', fontSize: 15 },
    itemMeta: { color: colors.textSecondary },
    itemDate: { color: colors.textSecondary, fontSize: 12 },
});
