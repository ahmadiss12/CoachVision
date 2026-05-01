import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { colors } from '@/src/theme/colors';
const minBmi = 10;
const maxBmi = 40;
function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}
function bmiCategory(bmi) {
    if (bmi < 18.5)
        return { label: 'Underweight', color: '#38BDF8' };
    if (bmi < 25)
        return { label: 'Normal', color: colors.success };
    if (bmi < 30)
        return { label: 'Overweight', color: colors.warning };
    return { label: 'Obese', color: colors.danger };
}
export function getBmiCategoryLabel(bmi) {
    if (bmi === null || !Number.isFinite(bmi))
        return '--';
    return bmiCategory(bmi).label;
}
export function BmiGauge({ bmi }) {
    const safeBmi = bmi && Number.isFinite(bmi) ? bmi : null;
    const currentCategory = safeBmi === null ? null : bmiCategory(safeBmi);
    const progress = safeBmi === null ? 0 : ((clamp(safeBmi, minBmi, maxBmi) - minBmi) / (maxBmi - minBmi)) * 100;
    return (<View style={styles.container}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>BMI scale</Text>
        <Text style={[styles.category, { color: currentCategory?.color ?? colors.textSecondary }]}>
          {currentCategory?.label ?? '--'}
        </Text>
      </View>

      <View style={styles.track}>
        <View style={[styles.segment, styles.segmentLow]}/>
        <View style={[styles.segment, styles.segmentNormal]}/>
        <View style={[styles.segment, styles.segmentHigh]}/>
        <View style={[styles.segment, styles.segmentVeryHigh]}/>
        <View style={[styles.marker, { left: `${progress}%` }]}/>
      </View>

      <View style={styles.labelsRow}>
        <Text style={styles.rangeText}>10</Text>
        <Text style={styles.rangeText}>18.5</Text>
        <Text style={styles.rangeText}>25</Text>
        <Text style={styles.rangeText}>30</Text>
        <Text style={styles.rangeText}>40</Text>
      </View>
    </View>);
}
const styles = StyleSheet.create({
    container: { gap: 8 },
    headerRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
    title: { color: colors.textSecondary, fontSize: 12, textTransform: 'uppercase' },
    category: { fontSize: 12, fontWeight: '700', textTransform: 'uppercase' },
    track: {
        position: 'relative',
        flexDirection: 'row',
        height: 14,
        borderRadius: 999,
        overflow: 'hidden',
        borderWidth: 1,
        borderColor: colors.border,
    },
    segment: { height: '100%' },
    segmentLow: { flex: 28.33, backgroundColor: '#38BDF8' },
    segmentNormal: { flex: 21.67, backgroundColor: colors.success },
    segmentHigh: { flex: 16.67, backgroundColor: colors.warning },
    segmentVeryHigh: { flex: 33.33, backgroundColor: colors.danger },
    marker: {
        position: 'absolute',
        top: -2,
        width: 4,
        height: 18,
        borderRadius: 2,
        backgroundColor: '#FFFFFF',
        marginLeft: -2,
    },
    labelsRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
    },
    rangeText: { color: colors.textSecondary, fontSize: 11 },
});
