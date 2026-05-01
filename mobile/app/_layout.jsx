import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import 'react-native-reanimated';
import { AppStateProvider, useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
function RootNavigator() {
    const { themeMode } = useAppState();
    const palette = getThemeColors(themeMode);
    return (<>
      <Stack screenOptions={{
            headerShown: false,
            // Small global offset so content doesn't touch status/notification area.
            contentStyle: { backgroundColor: palette.background, paddingTop: 4 },
        }}>
        <Stack.Screen name="index"/>
        <Stack.Screen name="(auth)"/>
        <Stack.Screen name="(app)"/>
      </Stack>
      <StatusBar style={themeMode === 'dark' ? 'light' : 'dark'}/>
    </>);
}
export default function RootLayout() {
    return (<AppStateProvider>
      <RootNavigator />
    </AppStateProvider>);
}
