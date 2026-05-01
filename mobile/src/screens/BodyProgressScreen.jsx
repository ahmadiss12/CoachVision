import React, { useMemo, useState } from 'react';
import { FlatList, SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { Card } from '@/src/components/Card';
import { Input } from '@/src/components/Input';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
function toNumber(text) {
    const value = Number(text);
    return Number.isFinite(value) ? value : 0;
}
export function BodyProgressScreen() {
    const { bodyMetrics, profile, addBodyMetricEntry, themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const [weightKg, setWeightKg] = useState(profile ? String(profile.weightKg) : '');
    const [bodyFatPercent, setBodyFatPercent] = useState(profile ? String(profile.bodyFatPercent) : '');
    const [date, setDate] = useState(new Date().toISOString().slice(0, 10));
    const latest = bodyMetrics[0];
    const oldest = bodyMetrics[bodyMetrics.length - 1];
    const deltaWeight = useMemo(() => {
        if (!latest || !oldest || bodyMetrics.length < 2)
            return 0;
        return Number((latest.weightKg - oldest.weightKg).toFixed(1));
    }, [latest, oldest, bodyMetrics.length]);
    const deltaBodyFat = useMemo(() => {
        if (!latest || !oldest || bodyMetrics.length < 2)
            return 0;
        return Number((latest.bodyFatPercent - oldest.bodyFatPercent).toFixed(1));
    }, [latest, oldest, bodyMetrics.length]);
    const weightColor = deltaWeight < 0 ? colors.success : deltaWeight > 0 ? colors.danger : colors.textSecondary;
    const bodyFatColor = deltaBodyFat < 0 ? colors.success : deltaBodyFat > 0 ? colors.danger : colors.textSecondary;
    const addEntry = () => {
        addBodyMetricEntry({
            weightKg: toNumber(weightKg),
            bodyFatPercent: toNumber(bodyFatPercent),
            date: date || undefined,
        });
    };
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <View style={styles.container}>
        <Text style={[styles.title, { color: colors.textPrimary }]}>Body Progress</Text>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
          Log weight and body fat to track your body changes.
        </Text>

        <Card style={styles.formCard}>
          <Input label="Date (YYYY-MM-DD)" value={date} onChangeText={setDate}/>
          <Input label="Weight (kg)" value={weightKg} keyboardType="numeric" placeholder="e.g. 72" onChangeText={setWeightKg}/>
          <Input label="Body fat (%)" value={bodyFatPercent} keyboardType="numeric" placeholder="e.g. 18" onChangeText={setBodyFatPercent}/>
          <PrimaryButton title="Add entry" onPress={addEntry}/>
        </Card>

        <Card style={styles.statsCard}>
          <Text style={[styles.statsTitle, { color: colors.textPrimary }]}>Change summary</Text>
          <Text style={[styles.statsText, { color: weightColor }]}>
            Weight change: {deltaWeight >= 0 ? '+' : ''}
            {deltaWeight} kg
          </Text>
          <Text style={[styles.statsText, { color: bodyFatColor }]}>
            Body fat change: {deltaBodyFat >= 0 ? '+' : ''}{deltaBodyFat}%
          </Text>
        </Card>

        <Text style={[styles.listTitle, { color: colors.textSecondary }]}>History</Text>
        {bodyMetrics.length === 0 ? (<Card>
            <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
              No entries yet. Add your first check-in above.
            </Text>
          </Card>) : (<FlatList data={bodyMetrics} keyExtractor={(item) => item.id} contentContainerStyle={styles.listContent} renderItem={({ item, index }) => {
                const previous = bodyMetrics[index + 1];
                const deltaWeightRow = previous ? Number((item.weightKg - previous.weightKg).toFixed(1)) : null;
                const deltaBodyFatRow = previous
                    ? Number((item.bodyFatPercent - previous.bodyFatPercent).toFixed(1))
                    : null;
                const deltaWeightRowColor = deltaWeightRow === null
                    ? colors.textSecondary
                    : deltaWeightRow < 0
                        ? colors.success
                        : deltaWeightRow > 0
                            ? colors.danger
                            : colors.textSecondary;
                const deltaBodyFatRowColor = deltaBodyFatRow === null
                    ? colors.textSecondary
                    : deltaBodyFatRow < 0
                        ? colors.success
                        : deltaBodyFatRow > 0
                            ? colors.danger
                            : colors.textSecondary;
                return (<Card style={[styles.itemCard, { borderColor: colors.border }]}>
                  <Text style={[styles.itemDate, { color: colors.textPrimary }]}>{item.date}</Text>
                  <Text style={[styles.itemMeta, { color: colors.textSecondary }]}>
                    {item.weightKg} kg • {item.bodyFatPercent}%
                  </Text>
                  <View style={styles.deltaRow}>
                    <Text style={[styles.deltaBadge, { color: deltaWeightRowColor, borderColor: deltaWeightRowColor }]}>
                      {deltaWeightRow === null
                        ? 'Weight Δ --'
                        : `Weight Δ ${deltaWeightRow >= 0 ? '+' : ''}${deltaWeightRow} kg`}
                    </Text>
                    <Text style={[styles.deltaBadge, { color: deltaBodyFatRowColor, borderColor: deltaBodyFatRowColor }]}>
                      {deltaBodyFatRow === null
                        ? 'Body Fat Δ --'
                        : `Body Fat Δ ${deltaBodyFatRow >= 0 ? '+' : ''}${deltaBodyFatRow}%`}
                    </Text>
                  </View>
                </Card>);
            }}/>)}
      </View>
    </SafeAreaView>);
}
const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { flex: 1, padding: 20, gap: 12 },
    title: { fontSize: 28, fontWeight: '700', marginTop: 10 },
    subtitle: { marginBottom: 6 },
    formCard: { gap: 10 },
    statsCard: { gap: 6 },
    statsTitle: { fontWeight: '700', fontSize: 16 },
    statsText: {},
    listTitle: { fontSize: 12, textTransform: 'uppercase', marginTop: 2 },
    emptyText: {},
    listContent: { gap: 8, paddingBottom: 24 },
    itemCard: { gap: 4 },
    itemDate: { fontWeight: '700' },
    itemMeta: {},
    deltaRow: { flexDirection: 'row', gap: 8, marginTop: 4, flexWrap: 'wrap' },
    deltaBadge: {
        borderWidth: 1,
        borderRadius: 999,
        paddingHorizontal: 8,
        paddingVertical: 4,
        fontSize: 12,
        fontWeight: '600',
    },
});
