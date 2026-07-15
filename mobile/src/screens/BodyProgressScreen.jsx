import { Ionicons } from '@expo/vector-icons';
import React, { useEffect, useMemo, useState } from 'react';
import { FlatList, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Card } from '@/src/components/Card';
import { Input } from '@/src/components/Input';
import { MetricTile } from '@/src/components/MetricTile';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

function toNumber(text) {
    const value = Number(text);
    return Number.isFinite(value) ? value : 0;
}

export function BodyProgressScreen() {
    const { bodyMetrics, profile, addBodyMetricEntry, loadBodyMetrics, themeMode } = useAppState();
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
    const addEntry = () => {
        addBodyMetricEntry({
            weightKg: toNumber(weightKg),
            bodyFatPercent: toNumber(bodyFatPercent),
            date: date || undefined,
        });
    };
    useEffect(() => {
        loadBodyMetrics();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <View style={styles.container}>
        <ScreenHeader showBack icon="analytics-outline" title="Body progress" subtitle="Weight and body composition check-ins." />

        <Card style={styles.formCard}>
          <Input icon="calendar-outline" label="Date" value={date} onChangeText={setDate}/>
          <Input icon="scale-outline" label="Weight (kg)" value={weightKg} keyboardType="numeric" placeholder="e.g. 72" onChangeText={setWeightKg}/>
          <Input icon="water-outline" label="Body fat (%)" value={bodyFatPercent} keyboardType="numeric" placeholder="e.g. 18" onChangeText={setBodyFatPercent}/>
          <PrimaryButton icon="add-circle-outline" title="Add entry" onPress={addEntry}/>
        </Card>

        <View style={styles.summaryGrid}>
          <MetricTile
            icon={deltaWeight < 0 ? 'trending-down-outline' : 'trending-up-outline'}
            label="Weight change"
            value={`${deltaWeight >= 0 ? '+' : ''}${deltaWeight} kg`}
            tone={deltaWeight < 0 ? 'success' : deltaWeight > 0 ? 'danger' : 'brand'}
          />
          <MetricTile
            icon={deltaBodyFat < 0 ? 'trending-down-outline' : 'trending-up-outline'}
            label="Body fat change"
            value={`${deltaBodyFat >= 0 ? '+' : ''}${deltaBodyFat}%`}
            tone={deltaBodyFat < 0 ? 'success' : deltaBodyFat > 0 ? 'danger' : 'brand'}
          />
        </View>

        <View style={styles.listHeader}>
          <Ionicons name="list-outline" size={18} color={colors.brandAlt} />
          <Text style={[styles.listTitle, { color: colors.textPrimary }]}>History</Text>
        </View>
        {bodyMetrics.length === 0 ? (<Card style={styles.emptyCard}>
            <Ionicons name="clipboard-outline" size={28} color={colors.textSecondary} />
            <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
              No entries yet. Add your first check-in above.
            </Text>
          </Card>) : (<FlatList
            data={bodyMetrics}
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.listContent}
            showsVerticalScrollIndicator={false}
            renderItem={({ item, index }) => {
                const previous = bodyMetrics[index + 1];
                const deltaWeightRow = previous ? Number((item.weightKg - previous.weightKg).toFixed(1)) : null;
                const deltaBodyFatRow = previous
                    ? Number((item.bodyFatPercent - previous.bodyFatPercent).toFixed(1))
                    : null;
                return (<ProgressItem
                  item={item}
                  deltaWeightRow={deltaWeightRow}
                  deltaBodyFatRow={deltaBodyFatRow}
                  colors={colors}
                />);
            }}
          />)}
      </View>
    </SafeAreaView>);
}

function ProgressItem({ item, deltaWeightRow, deltaBodyFatRow, colors }) {
    const weightColor = deltaWeightRow === null
        ? colors.textSecondary
        : deltaWeightRow < 0
            ? colors.success
            : deltaWeightRow > 0
                ? colors.danger
                : colors.textSecondary;
    const bodyFatColor = deltaBodyFatRow === null
        ? colors.textSecondary
        : deltaBodyFatRow < 0
            ? colors.success
            : deltaBodyFatRow > 0
                ? colors.danger
                : colors.textSecondary;
    return (
      <Card style={styles.itemCard}>
        <View style={styles.itemHeader}>
          <View style={[styles.itemIcon, { backgroundColor: colors.surfaceAlt }]}>
            <Ionicons name="calendar-outline" size={18} color={colors.brandAlt} />
          </View>
          <View style={styles.itemCopy}>
            <Text style={[styles.itemDate, { color: colors.textPrimary }]}>{item.date}</Text>
            <Text style={[styles.itemMeta, { color: colors.textSecondary }]}>
              {item.weightKg} kg | {item.bodyFatPercent}%
            </Text>
          </View>
        </View>
        <View style={styles.deltaRow}>
          <DeltaBadge color={weightColor} text={deltaWeightRow === null ? 'Weight --' : `Weight ${deltaWeightRow >= 0 ? '+' : ''}${deltaWeightRow} kg`} />
          <DeltaBadge color={bodyFatColor} text={deltaBodyFatRow === null ? 'Body fat --' : `Body fat ${deltaBodyFatRow >= 0 ? '+' : ''}${deltaBodyFatRow}%`} />
        </View>
      </Card>
    );
}

function DeltaBadge({ color, text }) {
    return (
      <Text style={[styles.deltaBadge, { color, borderColor: color }]}>{text}</Text>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { flex: 1, padding: 20, gap: 12 },
    formCard: { gap: 12 },
    summaryGrid: { flexDirection: 'row', gap: 10 },
    listHeader: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 2 },
    listTitle: { fontSize: 17, fontWeight: '900' },
    emptyCard: { alignItems: 'center', gap: 10 },
    emptyText: { textAlign: 'center', lineHeight: 20 },
    listContent: { gap: 8, paddingBottom: 24 },
    itemCard: { gap: 10 },
    itemHeader: { flexDirection: 'row', alignItems: 'center', gap: 10 },
    itemIcon: {
        width: 38,
        height: 38,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    itemCopy: { flex: 1, minWidth: 0 },
    itemDate: { fontWeight: '900' },
    itemMeta: { marginTop: 2, fontWeight: '700' },
    deltaRow: { flexDirection: 'row', gap: 8, flexWrap: 'wrap' },
    deltaBadge: {
        borderWidth: 1,
        borderRadius: 8,
        paddingHorizontal: 8,
        paddingVertical: 5,
        fontSize: 12,
        fontWeight: '900',
    },
});
