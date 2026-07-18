'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, buildWsUrl } from '@/lib/api';
import { useAuth } from '@/lib/auth-context';

export default function MessagesPage() {
  const { user } = useAuth();
  const [conversations, setConversations] = useState([]);
  const [activePartner, setActivePartner] = useState(null);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);
  const activePartnerRef = useRef(null);
  activePartnerRef.current = activePartner;

  const refreshConversations = useCallback(() => {
    api('/chat/conversations')
      .then(setConversations)
      .catch((err) => setError(err?.message || 'Failed to load conversations.'));
  }, []);

  useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

  const openConversation = useCallback((partner) => {
    setActivePartner(partner);
    setMessages([]);
    api(`/chat/${partner.partnerId}/messages`)
      .then((rows) => {
        setMessages(rows);
        refreshConversations(); // unread badges just cleared
      })
      .catch((err) => setError(err?.message || 'Failed to load messages.'));
  }, [refreshConversations]);

  // Live delivery: one chat socket for the whole page.
  useEffect(() => {
    const socket = new WebSocket(buildWsUrl('/ws/chat'));
    socket.onmessage = (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      if (message.type !== 'message') return;
      const partner = activePartnerRef.current;
      const belongsToOpenThread =
        partner &&
        (message.senderId === partner.partnerId || message.recipientId === partner.partnerId);
      if (belongsToOpenThread) {
        setMessages((prev) => (prev.some((m) => m.id === message.id) ? prev : [...prev, message]));
      }
      refreshConversations();
    };
    const keepAlive = setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) socket.send('ping');
    }, 25000);
    return () => {
      clearInterval(keepAlive);
      socket.close();
    };
  }, [refreshConversations]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async (event) => {
    event.preventDefault();
    const body = draft.trim();
    if (!body || !activePartner || isSending) return;
    setIsSending(true);
    try {
      const sent = await api(`/chat/${activePartner.partnerId}/messages`, {
        method: 'POST',
        body: { body },
      });
      setDraft('');
      setMessages((prev) => (prev.some((m) => m.id === sent.id) ? prev : [...prev, sent]));
      refreshConversations();
    } catch (err) {
      setError(err?.message || 'Send failed.');
    } finally {
      setIsSending(false);
    }
  };

  return (
    <>
      <h1 className="page-title">Messages</h1>
      <p className="page-subtitle">Chat with your linked clients — delivered instantly.</p>
      {error ? <div className="error-box">{error}</div> : null}

      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: 14, minHeight: 480 }}>
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          {conversations.length === 0 ? (
            <p className="empty">No conversations yet — link a client first.</p>
          ) : (
            conversations.map((conversation) => (
              <div
                key={conversation.partnerId}
                onClick={() => openConversation(conversation)}
                style={{
                  padding: '12px 14px',
                  cursor: 'pointer',
                  borderBottom: '1px solid var(--border)',
                  background:
                    activePartner?.partnerId === conversation.partnerId ? 'var(--surface-alt)' : 'transparent',
                }}
              >
                <div className="row between">
                  <b style={{ textTransform: 'capitalize' }}>{conversation.partnerName}</b>
                  {conversation.unreadCount > 0 ? (
                    <span className="badge danger">{conversation.unreadCount}</span>
                  ) : null}
                </div>
                <div className="muted" style={{ fontSize: 12, marginTop: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {conversation.lastMessage || 'No messages yet'}
                </div>
              </div>
            ))
          )}
        </div>

        <div className="card" style={{ display: 'flex', flexDirection: 'column', padding: 0 }}>
          {!activePartner ? (
            <p className="empty" style={{ margin: 'auto' }}>
              Select a conversation to start chatting.
            </p>
          ) : (
            <>
              <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)', fontWeight: 900, textTransform: 'capitalize' }}>
                {activePartner.partnerName}
                <span className="muted" style={{ fontWeight: 700 }}> · {activePartner.partnerRole}</span>
              </div>
              <div style={{ flex: 1, overflowY: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
                {messages.map((message) => {
                  const mine = message.senderId === user?.id;
                  return (
                    <div key={message.id} style={{ display: 'flex', justifyContent: mine ? 'flex-end' : 'flex-start' }}>
                      <div
                        style={{
                          maxWidth: '70%',
                          borderRadius: 12,
                          padding: '8px 13px',
                          background: mine ? 'var(--brand)' : 'var(--surface-alt)',
                          color: mine ? '#04222b' : 'var(--text)',
                          fontWeight: 600,
                          fontSize: 14,
                          lineHeight: 1.5,
                        }}
                      >
                        {message.body}
                        <div style={{ fontSize: 10, opacity: 0.65, textAlign: 'right', marginTop: 3 }}>
                          {new Date(message.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>
                    </div>
                  );
                })}
                <div ref={bottomRef} />
              </div>
              <form onSubmit={send} style={{ display: 'flex', gap: 8, padding: 12, borderTop: '1px solid var(--border)' }}>
                <input
                  className="input"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  placeholder="Write a message…"
                  maxLength={2000}
                />
                <button className="btn" type="submit" disabled={!draft.trim() || isSending}>
                  {isSending ? '…' : 'Send'}
                </button>
              </form>
            </>
          )}
        </div>
      </div>
    </>
  );
}
