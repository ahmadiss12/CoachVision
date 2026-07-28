import React, { useCallback } from 'react';
import { Alert, Pressable, Share, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as WebBrowser from 'expo-web-browser';
import { Card } from '@/src/components/Card';
import { apiConfig } from '@/src/services/api/config';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

/** Hostname only, so the URL stays readable on a narrow screen. */
function displayUrl(url) {
    return String(url || '').replace(/^https?:\/\//i, '');
}

/**
 * Points trainers at the web dashboard, which is where program building and
 * client review actually happen.
 *
 * `compact` renders the inline variant used inside the empty-clients state;
 * the default renders the standalone card used in Settings.
 */
export function TrainerDashboardCard({ compact = false }) {
    const { themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const url = apiConfig.webDashboardUrl;

    const openDashboard = useCallback(async () => {
        try {
            await WebBrowser.openBrowserAsync(url);
        } catch {
            Alert.alert('Could not open', `Visit ${url} in your browser.`);
        }
    }, [url]);

    // Trainers usually want this link on a computer, so sharing it to email or
    // chat is often more useful than opening it here on the phone.
    const shareDashboard = useCallback(async () => {
        try {
            await Share.share({
                message: `CoachVision trainer dashboard: ${url}`,
                url,
            });
        } catch {
            // User dismissed the share sheet.
        }
    }, [url]);

    if (compact) {
        return (
          <View style={styles.compactWrap}>
            <Text style={[styles.compactCopy, { color: colors.textSecondary }]}>
              Building programs and reviewing sessions is easier on the web dashboard.
            </Text>
            <View style={styles.actions}>
              <Pressable
                accessibilityRole="button"
                onPress={openDashboard}
                style={({ pressed }) => [
                    styles.primaryAction,
                    { backgroundColor: colors.brand, opacity: pressed ? 0.85 : 1 },
                ]}
              >
                <Ionicons name="open-outline" size={16} color="#FFFFFF" />
                <Text style={styles.primaryActionText}>Open dashboard</Text>
              </Pressable>
              <Pressable
                accessibilityRole="button"
                accessibilityLabel="Send the dashboard link"
                onPress={shareDashboard}
                style={({ pressed }) => [
                    styles.iconAction,
                    { borderColor: colors.border, opacity: pressed ? 0.85 : 1 },
                ]}
              >
                <Ionicons name="share-outline" size={17} color={colors.brandAlt} />
              </Pressable>
            </View>
          </View>
        );
    }

    return (
      <Card style={styles.card}>
        <View style={styles.header}>
          <View style={[styles.icon, { backgroundColor: `${colors.brandAlt}22` }]}>
            <Ionicons name="desktop-outline" size={20} color={colors.brandAlt} />
          </View>
          <View style={styles.headerCopy}>
            <Text style={[styles.title, { color: colors.textPrimary }]}>Trainer dashboard</Text>
            <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
              Build programs and review clients on a bigger screen
            </Text>
          </View>
        </View>

        <View style={[styles.urlPill, { backgroundColor: colors.surfaceAlt, borderColor: colors.border }]}>
          <Ionicons name="link-outline" size={15} color={colors.textSecondary} />
          <Text
            style={[styles.urlText, { color: colors.textSecondary }]}
            numberOfLines={1}
            ellipsizeMode="middle"
          >
            {displayUrl(url)}
          </Text>
        </View>

        <View style={styles.actions}>
          <Pressable
            accessibilityRole="button"
            onPress={openDashboard}
            style={({ pressed }) => [
                styles.primaryAction,
                { backgroundColor: colors.brand, opacity: pressed ? 0.85 : 1 },
            ]}
          >
            <Ionicons name="open-outline" size={16} color="#FFFFFF" />
            <Text style={styles.primaryActionText}>Open</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            onPress={shareDashboard}
            style={({ pressed }) => [
                styles.secondaryAction,
                { borderColor: colors.border, opacity: pressed ? 0.85 : 1 },
            ]}
          >
            <Ionicons name="share-outline" size={16} color={colors.brandAlt} />
            <Text style={[styles.secondaryActionText, { color: colors.textPrimary }]}>
              Send link
            </Text>
          </Pressable>
        </View>

        <Text style={[styles.hint, { color: colors.textSecondary }]}>
          Sign in with this same account.
        </Text>
      </Card>
    );
}

const styles = StyleSheet.create({
    card: { gap: 12 },
    header: { flexDirection: 'row', alignItems: 'center', gap: 12 },
    icon: {
        width: 42,
        height: 42,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    headerCopy: { flex: 1, minWidth: 0 },
    title: { fontSize: 15, fontWeight: '900' },
    subtitle: { fontSize: 12, marginTop: 2, fontWeight: '700', lineHeight: 16 },
    urlPill: {
        minHeight: 40,
        borderRadius: 8,
        borderWidth: 1,
        paddingHorizontal: 12,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    urlText: { flex: 1, fontSize: 12.5, fontWeight: '700' },
    actions: { flexDirection: 'row', gap: 8 },
    primaryAction: {
        flex: 1,
        minHeight: 44,
        borderRadius: 8,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 7,
    },
    primaryActionText: { color: '#FFFFFF', fontSize: 14, fontWeight: '800' },
    secondaryAction: {
        flex: 1,
        minHeight: 44,
        borderRadius: 8,
        borderWidth: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 7,
    },
    secondaryActionText: { fontSize: 14, fontWeight: '800' },
    hint: { fontSize: 11.5, fontWeight: '700', textAlign: 'center' },
    compactWrap: { width: '100%', gap: 10, marginTop: 4 },
    compactCopy: { fontSize: 13, lineHeight: 18, textAlign: 'center', fontWeight: '600' },
    iconAction: {
        width: 44,
        minHeight: 44,
        borderRadius: 8,
        borderWidth: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
});
