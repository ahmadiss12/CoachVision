import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { IconBadge } from '@/src/components/IconBadge';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

export function ScreenHeader({ title, subtitle, icon, eyebrow, right, showBack = false, backHref = '/(app)/(tabs)', onBackPress }) {
    const router = useRouter();
    const { themeMode } = useAppState();
    const palette = getThemeColors(themeMode);
    const handleBack = () => {
        if (onBackPress) {
            onBackPress();
            return;
        }
        if (typeof router.canGoBack === 'function' && router.canGoBack()) {
            router.back();
            return;
        }
        router.replace(backHref);
    };
    return (
      <View style={styles.header}>
        <View style={styles.titleRow}>
          {showBack ? (
            <Pressable
              accessibilityLabel="Go back"
              accessibilityRole="button"
              hitSlop={10}
              onPress={handleBack}
              style={({ pressed }) => [
                styles.backButton,
                {
                  backgroundColor: palette.surfaceAlt,
                  borderColor: palette.border,
                  opacity: pressed ? 0.78 : 1,
                },
              ]}
            >
              <Ionicons name="chevron-back" size={22} color={palette.brandAlt} />
            </Pressable>
          ) : null}
          {icon ? <IconBadge name={icon} /> : null}
          <View style={styles.copy}>
            {eyebrow ? <Text style={[styles.eyebrow, { color: palette.brandAlt }]}>{eyebrow}</Text> : null}
            <Text style={[styles.title, { color: palette.textPrimary }]}>{title}</Text>
          </View>
          {right}
        </View>
        {subtitle ? <Text style={[styles.subtitle, { color: palette.textSecondary }]}>{subtitle}</Text> : null}
      </View>
    );
}

const styles = StyleSheet.create({
    header: {
        gap: 8,
        marginTop: 8,
    },
    titleRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
    },
    copy: {
        flex: 1,
        minWidth: 0,
    },
    backButton: {
        width: 42,
        height: 42,
        borderRadius: 8,
        borderWidth: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    eyebrow: {
        fontSize: 12,
        fontWeight: '900',
        textTransform: 'uppercase',
        marginBottom: 2,
    },
    title: {
        fontSize: 29,
        lineHeight: 34,
        fontWeight: '900',
    },
    subtitle: {
        fontSize: 15,
        lineHeight: 21,
    },
});
