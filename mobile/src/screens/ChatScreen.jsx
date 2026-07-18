import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams } from 'expo-router';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { ScreenHeader } from '@/src/components/ScreenHeader';
import { buildChatSocketUrl, getMessages, sendMessage } from '@/src/services/api/chat';
import { useAppState } from '@/src/state/app-state';
import { getThemeColors } from '@/src/theme/colors';

export function ChatScreen() {
    const { partnerId, partnerName } = useLocalSearchParams();
    const { authUser, authTokens, ensureFreshAccessToken, themeMode } = useAppState();
    const colors = getThemeColors(themeMode);
    const [messages, setMessages] = useState([]);
    const [draft, setDraft] = useState('');
    const [isSending, setIsSending] = useState(false);
    const socketRef = useRef(null);

    const load = useCallback(async () => {
        if (!partnerId) return;
        try {
            setMessages(await getMessages(partnerId));
        } catch (err) {
            if (err?.status === 401 && (await ensureFreshAccessToken())) {
                try {
                    setMessages(await getMessages(partnerId));
                } catch {
                    // empty thread shown
                }
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [partnerId]);

    useEffect(() => {
        load();
    }, [load]);

    // Live delivery over WebSocket, with a slow poll as a safety net.
    useEffect(() => {
        if (!partnerId || !authTokens?.accessToken) return undefined;
        let socket;
        try {
            socket = new WebSocket(buildChatSocketUrl(authTokens.accessToken));
            socketRef.current = socket;
            socket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    if (
                        message.type === 'message' &&
                        (message.senderId === partnerId || message.recipientId === partnerId)
                    ) {
                        setMessages((prev) =>
                            prev.some((m) => m.id === message.id) ? prev : [...prev, message],
                        );
                    }
                } catch {
                    // ignore malformed frames
                }
            };
        } catch {
            socket = null;
        }
        const poll = setInterval(load, 15000);
        return () => {
            clearInterval(poll);
            if (socket) socket.close();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [partnerId, authTokens?.accessToken]);

    const submit = async () => {
        const body = draft.trim();
        if (!body || isSending) return;
        setIsSending(true);
        try {
            const sent = await sendMessage(partnerId, body);
            setDraft('');
            setMessages((prev) => (prev.some((m) => m.id === sent.id) ? prev : [...prev, sent]));
        } catch {
            // keep draft so the user can retry
        } finally {
            setIsSending(false);
        }
    };

    const myId = authUser?.id;

    return (<SafeAreaView style={[styles.safe, { backgroundColor: colors.background }]} edges={['top', 'left', 'right']}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={0}
      >
        <View style={styles.container}>
          <ScreenHeader
            showBack
            icon="chatbubbles-outline"
            title={partnerName || 'Chat'}
            subtitle="Messages stay between you and your coach."
          />
          <FlatList
            data={[...messages].reverse()}
            inverted
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.list}
            showsVerticalScrollIndicator={false}
            renderItem={({ item }) => {
                const mine = item.senderId === myId;
                return (
                  <View style={[styles.bubbleRow, mine ? styles.mineRow : styles.theirsRow]}>
                    <View
                      style={[
                        styles.bubble,
                        mine
                          ? { backgroundColor: colors.brandAlt }
                          : { backgroundColor: colors.surface, borderColor: colors.border, borderWidth: 1 },
                      ]}
                    >
                      <Text style={[styles.bubbleText, { color: mine ? '#04222B' : colors.textPrimary }]}>
                        {item.body}
                      </Text>
                      <Text style={[styles.bubbleTime, { color: mine ? '#04222B99' : colors.textSecondary }]}>
                        {new Date(item.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </Text>
                    </View>
                  </View>
                );
            }}
            ListEmptyComponent={
              <Text style={[styles.empty, { color: colors.textSecondary }]}>
                Say hello — this is the start of your conversation.
              </Text>
            }
          />
          <View style={[styles.inputRow, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <TextInput
              style={[styles.input, { color: colors.textPrimary }]}
              value={draft}
              onChangeText={setDraft}
              placeholder="Write a message…"
              placeholderTextColor={colors.textSecondary}
              multiline
              maxLength={2000}
            />
            <Pressable
              onPress={submit}
              disabled={!draft.trim() || isSending}
              style={[
                styles.sendButton,
                { backgroundColor: draft.trim() && !isSending ? colors.brandAlt : colors.surfaceAlt },
              ]}
            >
              <Ionicons name="send" size={18} color={draft.trim() && !isSending ? '#04222B' : colors.textSecondary} />
            </Pressable>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>);
}

const styles = StyleSheet.create({
    safe: { flex: 1 },
    flex: { flex: 1 },
    container: { flex: 1, padding: 20, gap: 10 },
    list: { gap: 8, paddingVertical: 8 },
    bubbleRow: { flexDirection: 'row' },
    mineRow: { justifyContent: 'flex-end' },
    theirsRow: { justifyContent: 'flex-start' },
    bubble: {
        maxWidth: '80%',
        borderRadius: 14,
        paddingHorizontal: 13,
        paddingVertical: 9,
        gap: 3,
    },
    bubbleText: { fontSize: 15, lineHeight: 20, fontWeight: '600' },
    bubbleTime: { fontSize: 10, fontWeight: '700', alignSelf: 'flex-end' },
    empty: { textAlign: 'center', transform: [{ scaleY: -1 }], paddingVertical: 30 },
    inputRow: {
        flexDirection: 'row',
        alignItems: 'flex-end',
        gap: 8,
        borderWidth: 1,
        borderRadius: 14,
        paddingHorizontal: 12,
        paddingVertical: 8,
    },
    input: { flex: 1, fontSize: 15, maxHeight: 110, paddingTop: 6 },
    sendButton: {
        width: 38,
        height: 38,
        borderRadius: 19,
        alignItems: 'center',
        justifyContent: 'center',
    },
});
