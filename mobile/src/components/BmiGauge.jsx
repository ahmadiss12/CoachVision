import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

const minBmi = 10;
const maxBmi = 40;

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function bmiCategory(bmi, colors) {
    if (bmi < 18.5)
        return { label: 'Underweight', color: colors.info };
    if (bmi < 25)
        return { label: 'Normal', color: colors.success };
    if (bmi < 30)
        return { label: 'Overweight', color: colors.warning };
    return { label: 'Obese', color: colors.danger };
}

export function getBmiCategoryLabel(bmi) {
    if (bmi === null || !Number.isFinite(bmi))
        return '--';
    return bmiCategory(bmi, getThemeColors('dark')).label;
}

export function BmiGauge({ bmi }) {
    const { themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const safeBmi = bmi && Number.isFinite(bmi) ? bmi : null;
    const currentCategory = safeBmi === null ? null : bmiCategory(safeBmi, colors);
    const progress = safeBmi === null ? 0 : ((clamp(safeBmi, minBmi, maxBmi) - minBmi) / (maxBmi - minBmi)) * 100;
    return (<View style={styles.container}>
      <View style={styles.headerRow}>
        <Text style={[styles.title, { color: colors.textSecondary }]}>BMI scale</Text>
        <Text style={[styles.category, { color: currentCategory?.color ?? colors.textSecondary }]}>
          {currentCategory?.label ?? '--'}
        </Text>
      </View>

      <View style={[styles.track, { borderColor: colors.border }]}>
        <View style={[styles.segment, { flex: 28.33, backgroundColor: colors.info }]}/>
        <View style={[styles.segment, { flex: 21.67, backgroundColor: colors.success }]}/>
        <View style={[styles.segment, { flex: 16.67, backgroundColor: colors.warning }]}/>
        <View style={[styles.segment, { flex: 33.33, backgroundColor: colors.danger }]}/>
        <View style={[styles.marker, { left: `${progress}%` }]}/>
      </View>

      <View style={styles.labelsRow}>
        {['10', '18.5', '25', '30', '40'].map((label) => (
          <Text key={label} style={[styles.rangeText, { color: colors.textSecondary }]}>{label}</Text>
        ))}
      </View>
    </View>);
}

const styles = StyleSheet.create({
    container: { gap: 8 },
    headerRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
    title: { fontSize: 12, textTransform: 'uppercase', fontWeight: '800' },
    category: { fontSize: 12, fontWeight: '900', textTransform: 'uppercase' },
    track: {
        position: 'relative',
        flexDirection: 'row',
        height: 14,
        borderRadius: 8,
        overflow: 'hidden',
        borderWidth: 1,
    },
    segment: { height: '100%' },
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
    rangeText: { fontSize: 11 },
});
