import { Redirect, Stack, useSegments } from 'expo-router';
import { useAppState } from '@/src/state/app-state';
export default function AppLayout() {
    const segments = useSegments();
    const { isAuthenticated, needsOnboarding, needsGoalsOnboarding } = useAppState();
    if (!isAuthenticated) {
        return <Redirect href="/(auth)/login"/>;
    }
    const currentLeaf = segments[segments.length - 1];
    if (needsOnboarding && currentLeaf !== 'onboarding-profile') {
        return <Redirect href="/(app)/onboarding-profile"/>;
    }
    if (!needsOnboarding && needsGoalsOnboarding && currentLeaf !== 'onboarding-goals') {
        return <Redirect href="/(app)/onboarding-goals"/>;
    }
    return (<Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="(tabs)"/>
      <Stack.Screen name="onboarding-profile"/>
      <Stack.Screen name="onboarding-goals"/>
      <Stack.Screen name="profile"/>
      <Stack.Screen name="goals"/>
      <Stack.Screen name="body-progress"/>
      <Stack.Screen name="workout-setup"/>
      <Stack.Screen name="workout-live"/>
      <Stack.Screen name="session-summary"/>
      <Stack.Screen name="history"/>
    </Stack>);
}
