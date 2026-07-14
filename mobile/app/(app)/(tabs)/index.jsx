import { Redirect } from 'expo-router';
import { HomeScreen } from '@/src/screens/HomeScreen';
import { useAppState } from '@/src/state/app-state';

export default function HomeTabRoute() {
    const { userRole } = useAppState();
    if (userRole === 'trainer') {
        return <Redirect href="/(app)/(tabs)/clients" />;
    }
    if (userRole === 'admin') {
        return <Redirect href="/(app)/(tabs)/users" />;
    }
    return <HomeScreen />;
}
