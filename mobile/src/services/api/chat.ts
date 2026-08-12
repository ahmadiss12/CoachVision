import { apiRequest, type Schemas } from './client';
import { buildWsUrl } from './config';

export type Conversation = Schemas['ConversationResponse'];
export type ChatMessage = Schemas['ChatMessageResponse'];

/** `GET /chat/unread-count` returns a free-form object, not a named schema. */
export type UnreadCount = { count?: number } & Record<string, unknown>;

export async function listConversations(): Promise<Conversation[]> {
  return apiRequest<Conversation[]>('/chat/conversations', { auth: true });
}

export async function getUnreadCount(): Promise<UnreadCount> {
  return apiRequest<UnreadCount>('/chat/unread-count', { auth: true });
}

export async function getMessages(partnerId: string, limit = 50): Promise<ChatMessage[]> {
  return apiRequest<ChatMessage[]>(`/chat/${partnerId}/messages?limit=${limit}`, {
    auth: true,
  });
}

export async function sendMessage(partnerId: string, body: string): Promise<ChatMessage> {
  const payload: Schemas['ChatSendRequest'] = { body };
  return apiRequest<ChatMessage>(`/chat/${partnerId}/messages`, {
    method: 'POST',
    auth: true,
    body: payload,
  });
}

export function buildChatSocketUrl(accessToken: string): string {
  return buildWsUrl('/ws/chat', { token: accessToken });
}
