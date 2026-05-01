import { Ionicons } from '@expo/vector-icons';
import { Tabs } from 'expo-router';
import { colors } from '@/src/theme/colors';
export default function TabsLayout() {
    return (<Tabs screenOptions={{
            headerShown: false,
            tabBarStyle: {
                backgroundColor: colors.surface,
                borderTopColor: colors.border,
                height: 64,
                paddingBottom: 8,
                paddingTop: 8,
            },
            tabBarActiveTintColor: colors.brandAlt,
            tabBarInactiveTintColor: colors.textSecondary,
        }}>
      <Tabs.Screen name="index" options={{
            title: 'Home',
            tabBarIcon: ({ color, size }) => <Ionicons name="home" color={color} size={size}/>,
        }}/>
      <Tabs.Screen name="workout" options={{
            title: 'Workout',
            tabBarIcon: ({ color, size }) => <Ionicons name="barbell" color={color} size={size}/>,
        }}/>
      <Tabs.Screen name="settings" options={{
            title: 'Settings',
            tabBarIcon: ({ color, size }) => <Ionicons name="settings" color={color} size={size}/>,
        }}/>
      <Tabs.Screen name="profile" options={{
            title: 'Profile',
            tabBarIcon: ({ color, size }) => <Ionicons name="person-circle" color={color} size={size}/>,
        }}/>
    </Tabs>);
}
