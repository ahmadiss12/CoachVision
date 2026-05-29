import { Ionicons } from '@expo/vector-icons';
import { Tabs } from 'expo-router';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
export default function TabsLayout() {
    const { themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    return (<Tabs screenOptions={{
            headerShown: false,
            tabBarStyle: {
                backgroundColor: colors.surface,
                borderTopColor: colors.border,
                height: 68,
                paddingBottom: 8,
                paddingTop: 8,
            },
            tabBarActiveTintColor: colors.brandAlt,
            tabBarInactiveTintColor: colors.textSecondary,
            tabBarLabelStyle: {
                fontWeight: '800',
                fontSize: 11,
            },
        }}>
      <Tabs.Screen name="index" options={{
            title: 'Home',
            tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'home' : 'home-outline'} color={color} size={22}/>,
        }}/>
      <Tabs.Screen name="workout" options={{
            title: 'Workout',
            tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'barbell' : 'barbell-outline'} color={color} size={22}/>,
        }}/>
      <Tabs.Screen name="settings" options={{
            title: 'Settings',
            tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'options' : 'options-outline'} color={color} size={22}/>,
        }}/>
      <Tabs.Screen name="profile" options={{
            title: 'Profile',
            tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'person-circle' : 'person-circle-outline'} color={color} size={23}/>,
        }}/>
    </Tabs>);
}
