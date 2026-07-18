import { apiRequest } from './client';
import { buildWsUrl } from './config';

export async function listConversations() {
  return apiRequest('/chat/conversations', { auth: true });
}

export async function getUnreadCount() {
  return apiRequest('/chat/unread-count', { auth: true });
}

export async function getMessages(partnerId, limit = 50) {
  return apiRequest(`/chat/${partnerId}/messages?limit=${limit}`, { auth: true });
}

export async function sendMessage(partnerId, body) {
  return apiRequest(`/chat/${partnerId}/messages`, {
    method: 'POST',
    auth: true,
    body: { body },
  });
}

export function buildChatSocketUrl(accessToken) {
  return buildWsUrl('/ws/chat', { token: accessToken });
}
