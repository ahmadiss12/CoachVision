import { Redirect, Stack } from 'expo-router';
import { useAppState } from '@/src/state/app-state';
export default function AuthLayout() {
    const { isAuthenticated, needsOnboarding, needsGoalsOnboarding } = useAppState();
    if (isAuthenticated) {
        if (needsOnboarding)
            return <Redirect href="/(app)/onboarding-profile"/>;
        if (needsGoalsOnboarding)
            return <Redirect href="/(app)/onboarding-goals"/>;
        return <Redirect href="/(app)/(tabs)"/>;
    }
    return <Stack screenOptions={{ headerShown: false }}/>;
}
