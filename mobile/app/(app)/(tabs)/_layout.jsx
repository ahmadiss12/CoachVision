import { Ionicons } from '@expo/vector-icons';
import { Tabs } from 'expo-router';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';
export default function TabsLayout() {
    const { themeMode, userRole } = useAppState();
    const colors = getThemeColors(themeMode);
    const isClient = userRole === 'client';
    const isTrainer = userRole === 'trainer';
    const isAdmin = userRole === 'admin';
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
            href: isClient ? undefined : null,
            tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'home' : 'home-outline'} color={color} size={22}/>,
        }}/>
      <Tabs.Screen name="workout" options={{
            title: 'Workout',
            href: isClient ? undefined : null,
            tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'barbell' : 'barbell-outline'} color={color} size={22}/>,
        }}/>
      <Tabs.Screen name="clients" options={{
            title: 'Clients',
            href: isTrainer ? undefined : null,
            tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'people' : 'people-outline'} color={color} size={22}/>,
        }}/>
      <Tabs.Screen name="invites" options={{
            title: 'Invites',
            href: isTrainer ? undefined : null,
            tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'mail-open' : 'mail-open-outline'} color={color} size={22}/>,
        }}/>
      <Tabs.Screen name="users" options={{
            title: 'Users',
            href: isAdmin ? undefined : null,
            tabBarIcon: ({ color, focused }) => <Ionicons name={focused ? 'shield-checkmark' : 'shield-checkmark-outline'} color={color} size={22}/>,
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
