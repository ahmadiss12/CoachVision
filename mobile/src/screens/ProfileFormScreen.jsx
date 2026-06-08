import DateTimePicker, { DateTimePickerAndroid } from '@react-native-community/datetimepicker';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { createElement, useMemo, useState } from 'react';
import { Modal, Platform, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';
import { BmiGauge, getBmiCategoryLabel } from '@/src/components/BmiGauge';
import { Card } from '@/src/components/Card';
import { Input } from '@/src/components/Input';
import { PrimaryButton } from '@/src/components/PrimaryButton';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

function toNumber(text) {
    const value = Number(text);
    return Number.isFinite(value) ? value : 0;
}

function parseYmdToDate(ymd) {
    const parts = ymd.split('-').map(Number);
    const y = parts[0];
    const m = parts[1];
    const d = parts[2];
    if (!y || !m || !d)
        return new Date(2000, 0, 1);
    return new Date(y, m - 1, d);
}

function formatYmd(d) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
}

export function ProfileFormScreen({ mode }) {
    const router = useRouter();
    const { profile, saveProfile, themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const [dobDate, setDobDate] = useState(() => profile?.dateOfBirth ? parseYmdToDate(profile.dateOfBirth) : new Date(2000, 0, 1));
    const [iosPickerVisible, setIosPickerVisible] = useState(false);
    const [heightCm, setHeightCm] = useState(profile?.heightCm ? String(profile.heightCm) : '');
    const [weightKg, setWeightKg] = useState(profile?.weightKg ? String(profile.weightKg) : '');
    const [bodyFatPercent, setBodyFatPercent] = useState(profile?.bodyFatPercent ? String(profile.bodyFatPercent) : '');
    const [formError, setFormError] = useState(null);
    const bmiPreview = useMemo(() => {
        const h = toNumber(heightCm);
        const w = toNumber(weightKg);
        if (!h || !w)
            return '--';
        const meters = h / 100;
        const bmi = w / (meters * meters);
        if (!Number.isFinite(bmi))
            return '--';
        return bmi.toFixed(1);
    }, [heightCm, weightKg]);
    const bmiNumber = bmiPreview === '--' ? null : Number(bmiPreview);
    const bmiCategory = getBmiCategoryLabel(bmiNumber);
    const maximumDob = useMemo(() => new Date(), []);

    const openDobPicker = () => {
        if (Platform.OS === 'android') {
            DateTimePickerAndroid.open({
                value: dobDate,
                mode: 'date',
                maximumDate: maximumDob,
                onChange: (_event, date) => {
                    if (date) {
                        setDobDate(date);
                    }
                },
            });
            return;
        }
        if (Platform.OS === 'ios') {
            setIosPickerVisible(true);
        }
    };

    const handleSave = () => {
        const parsedHeight = toNumber(heightCm);
        const parsedWeight = toNumber(weightKg);
        const parsedBodyFat = toNumber(bodyFatPercent);
        const dobString = formatYmd(dobDate);
        if (Number.isNaN(dobDate.getTime())) {
            setFormError('Please choose a valid date of birth.');
            return;
        }
        if (heightCm && (parsedHeight < 80 || parsedHeight > 260)) {
            setFormError('Height looks invalid. Use centimeters (80-260).');
            return;
        }
        if (weightKg && (parsedWeight < 20 || parsedWeight > 400)) {
            setFormError('Weight looks invalid. Use kilograms (20-400).');
            return;
        }
        if (bodyFatPercent && (parsedBodyFat < 0 || parsedBodyFat > 70)) {
            setFormError('Body fat should be between 0% and 70%.');
            return;
        }
        setFormError(null);
        saveProfile({
            dateOfBirth: dobString,
            heightCm: parsedHeight,
            weightKg: parsedWeight,
            bodyFatPercent: parsedBodyFat,
        });
        if (mode === 'onboarding') {
            router.push('/(app)/onboarding-goals');
        }
        else {
            router.back();
        }
    };

    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
        <ScreenHeader
          showBack={mode !== 'onboarding'}
          icon="person-circle-outline"
          title={mode === 'onboarding' ? 'Complete profile' : 'Edit body profile'}
          subtitle={mode === 'onboarding'
            ? 'Body data personalizes your training insights.'
            : 'Update your latest measurements at any time.'}
        />

        <Card style={styles.formCard}>
          <View style={styles.dobBlock}>
            <Text style={[styles.dobLabel, { color: colors.textSecondary }]}>Date of birth</Text>
            {Platform.OS === 'web' ? (
              createElement('input', {
                  type: 'date',
                  value: formatYmd(dobDate),
                  max: formatYmd(maximumDob),
                  onChange: (e) => {
                      const v = e.currentTarget.value;
                      if (v) {
                          const [y, m, d] = v.split('-').map(Number);
                          if (y && m && d)
                              setDobDate(new Date(y, m - 1, d));
                      }
                  },
                  style: {
                      width: '100%',
                      height: 48,
                      borderRadius: 8,
                      borderWidth: 1,
                      borderStyle: 'solid',
                      borderColor: colors.border,
                      backgroundColor: colors.surface,
                      color: colors.textPrimary,
                      paddingLeft: 14,
                      paddingRight: 14,
                      fontSize: 16,
                      boxSizing: 'border-box',
                  },
              })
            ) : (<Pressable onPress={openDobPicker} style={({ pressed }) => [
                    styles.dobField,
                    {
                        borderColor: colors.border,
                        backgroundColor: colors.surface,
                        opacity: pressed ? 0.9 : 1,
                    },
                ]}>
                <Ionicons name="calendar-outline" size={18} color={colors.brandAlt} />
                <Text style={[styles.dobValue, { color: colors.textPrimary }]}>{formatYmd(dobDate)}</Text>
                <Ionicons name="chevron-down" size={18} color={colors.textSecondary} />
              </Pressable>)}
          </View>

          <Input icon="resize-outline" label="Height (cm)" value={heightCm} keyboardType="numeric" placeholder="e.g. 175" onChangeText={setHeightCm}/>
          <Input icon="scale-outline" label="Weight (kg)" value={weightKg} keyboardType="numeric" placeholder="e.g. 72" onChangeText={setWeightKg}/>
          <Input icon="water-outline" label="Body fat (%)" value={bodyFatPercent} keyboardType="numeric" placeholder="e.g. 18" onChangeText={setBodyFatPercent}/>
        </Card>

        {Platform.OS === 'ios' ? (<Modal animationType="slide" transparent visible={iosPickerVisible} onRequestClose={() => setIosPickerVisible(false)}>
            <Pressable style={styles.modalBackdrop} onPress={() => setIosPickerVisible(false)}>
              <Pressable style={[styles.modalSheet, { backgroundColor: colors.surface }]} onPress={(e) => e.stopPropagation()}>
                <View style={[styles.modalHeader, { borderBottomColor: colors.border }]}>
                  <Pressable onPress={() => setIosPickerVisible(false)} hitSlop={12}>
                    <Text style={[styles.modalAction, { color: colors.textSecondary }]}>Cancel</Text>
                  </Pressable>
                  <Text style={[styles.modalTitle, { color: colors.textPrimary }]}>Date of birth</Text>
                  <Pressable onPress={() => setIosPickerVisible(false)} hitSlop={12}>
                    <Text style={[styles.modalAction, { color: colors.brandAlt, fontWeight: '700' }]}>Done</Text>
                  </Pressable>
                </View>
                <DateTimePicker
                  value={dobDate}
                  mode="date"
                  display="spinner"
                  themeVariant={themeMode === 'dark' ? 'dark' : 'light'}
                  maximumDate={maximumDob}
                  onChange={(_, date) => {
                      if (date)
                          setDobDate(date);
                  }}
                />
              </Pressable>
            </Pressable>
          </Modal>) : null}

        <View style={[
            styles.bmiBox,
            { borderColor: colors.border, backgroundColor: colors.surface },
        ]}>
          <View style={styles.bmiHeader}>
            <Ionicons name="speedometer-outline" size={18} color={colors.brandAlt} />
            <Text style={[styles.bmiLabel, { color: colors.textSecondary }]}>BMI preview</Text>
          </View>
          <Text style={[styles.bmiValue, { color: colors.textPrimary }]}>{bmiPreview}</Text>
          <Text style={[styles.bmiCategory, { color: colors.textSecondary }]}>Category: {bmiCategory}</Text>
          <BmiGauge bmi={bmiNumber}/>
        </View>

        {formError ? <Text style={[styles.errorText, { color: colors.danger }]}>{formError}</Text> : null}

        <PrimaryButton icon="checkmark-circle-outline" title={mode === 'onboarding' ? 'Save and continue' : 'Save changes'} onPress={handleSave} disabled={false}/>
      </ScrollView>
    </SafeAreaView>);
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    container: { padding: 20, gap: 14, paddingBottom: 32 },
    formCard: { gap: 12 },
    bmiBox: {
        marginTop: 4,
        borderWidth: 1,
        borderRadius: 8,
        padding: 12,
    },
    bmiHeader: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    bmiLabel: { fontSize: 12, textTransform: 'uppercase', fontWeight: '900' },
    bmiValue: { fontSize: 30, fontWeight: '900', marginTop: 6 },
    bmiCategory: { marginTop: 4, marginBottom: 4 },
    errorText: { fontSize: 13, fontWeight: '700' },
    dobBlock: { gap: 8 },
    dobLabel: { fontSize: 13, fontWeight: '700' },
    dobField: {
        minHeight: 48,
        borderWidth: 1,
        borderRadius: 8,
        paddingHorizontal: 14,
        paddingVertical: 12,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
    },
    dobValue: { fontSize: 16, fontWeight: '700', flex: 1 },
    modalBackdrop: {
        flex: 1,
        justifyContent: 'flex-end',
        backgroundColor: 'rgba(0,0,0,0.45)',
    },
    modalSheet: {
        borderTopLeftRadius: 16,
        borderTopRightRadius: 16,
        paddingBottom: 24,
        overflow: 'hidden',
    },
    modalHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderBottomWidth: StyleSheet.hairlineWidth,
    },
    modalTitle: { fontSize: 16, fontWeight: '700' },
    modalAction: { fontSize: 16 },
});
