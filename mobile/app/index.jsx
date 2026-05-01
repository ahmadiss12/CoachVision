import { Redirect } from 'expo-router';
import { useAppState } from '@/src/state/app-state';
export default function IndexRoute() {
    const { isAuthenticated, needsOnboarding, needsGoalsOnboarding } = useAppState();
    if (!isAuthenticated)
        return <Redirect href="/(auth)/welcome"/>;
    if (needsOnboarding)
        return <Redirect href="/(app)/onboarding-profile"/>;
    if (needsGoalsOnboarding)
        return <Redirect href="/(app)/onboarding-goals"/>;
    return <Redirect href="/(app)/(tabs)"/>;
}
