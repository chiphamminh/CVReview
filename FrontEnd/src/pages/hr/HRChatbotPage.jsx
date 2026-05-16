import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  Layout, Menu, Input, Button, Typography, Space,
  Segmented, Avatar, Select, Modal, Spin, Tooltip, Tag,
  message as antMessage,
} from 'antd';
import { SendOutlined, RobotOutlined, UserOutlined, PlusOutlined, MenuFoldOutlined, MenuUnfoldOutlined, ReloadOutlined } from '@ant-design/icons';
import ChatMarkdown from '@/components/chatbot/ChatMarkdown';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { positionApi } from '@/api/position.api';
import { chatbotApi } from '@/api/chatbot.api';
import useAuthStore from '@/store/authStore';
import useChatbotStore from '@/store/chatbotStore';

const { Sider, Content } = Layout;
const { Title, Text } = Typography;

const CHAT_PAGE_SIZE = 10;

const formatDate = (dateStr) => {
  const d = new Date(dateStr);
  return `${String(d.getDate()).padStart(2, '0')}-${String(d.getMonth() + 1).padStart(2, '0')}-${d.getFullYear()}`;
};

const mapRole = (role) => (role === 'USER' ? 'user' : 'assistant');

const HRChatbotPage = () => {
  const { positionId: urlPositionId } = useParams();
  const navigate = useNavigate();
  const { user } = useAuthStore();

  const [sessions, setSessions] = useState([]);
  const {
    selectedPositionId: storedPositionId,
    currentSessionId,
    mode,
    sidebarCollapsed,
    setSelectedPositionId,
    setCurrentSessionId,
    setMode,
    setSidebarCollapsed,
  } = useChatbotStore();
  const selectedPositionId = urlPositionId ? parseInt(urlPositionId) : storedPositionId;

  // Sync URL position → Zustand so it survives back-navigation to /hr/chatbot (no ID in URL)
  useEffect(() => {
    if (urlPositionId) setSelectedPositionId(parseInt(urlPositionId));
  }, [urlPositionId]); // eslint-disable-line react-hooks/exhaustive-deps

  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const streamBufferRef = useRef('');
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [hasOlderMessages, setHasOlderMessages] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [firstMessageId, setFirstMessageId] = useState(null);

  const [showModeModal, setShowModeModal] = useState(false);
  const [modalMode, setModalMode] = useState('Internal');
  const [creatingSession, setCreatingSession] = useState(false);

  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const topSentinelRef = useRef(null);
  const suppressScrollRef = useRef(false);
  // Captures the session that was active when this page mounted (for restore on back-navigation)
  const initialSessionId = useRef(currentSessionId);

  // ── Active positions for dropdown + session title resolution ──
  const { data: positionsData } = useQuery({
    queryKey: ['activePositions'],
    queryFn: () => positionApi.filter({ isActive: true, size: 50 }).then(r => r.data?.content || []),
  });
  const positions = positionsData || [];

  const positionMap = useMemo(() => {
    const map = {};
    positions.forEach(p => { map[p.id] = p; });
    return map;
  }, [positions]);

  const getSessionTitle = useCallback((session) => {
    const pos = positionMap[session.positionId];
    const date = formatDate(session.createdAt);
    if (!pos) return `Session — ${date}`;
    return `${pos.seniority} ${pos.title} — ${date}`;
  }, [positionMap]);

  // ── Session restore — reload history when navigating back to this page ──
  useEffect(() => {
    const sessionId = initialSessionId.current;
    if (!sessionId) return;
    setLoadingHistory(true);
    setMessages([]);
    setFirstMessageId(null);
    setHasOlderMessages(false);
    chatbotApi.getSessionHistory(sessionId, { limit: CHAT_PAGE_SIZE })
      .then(res => {
        const history = res.data || [];
        const feMsgs = history.map(m => ({ id: m.id, role: mapRole(m.role), content: m.content }));
        setMessages(feMsgs);
        if (history.length > 0) {
          setFirstMessageId(history[0].id);
          setHasOlderMessages(history.length === CHAT_PAGE_SIZE);
        }
      })
      .catch(() => { setCurrentSessionId(null); })
      .finally(() => { setLoadingHistory(false); });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Load sessions for current position only ──
  const loadSessions = useCallback(async () => {
    if (!selectedPositionId) {
      setSessions([]);
      return;
    }
    try {
      const res = await chatbotApi.getSessions({ page: 0, size: 20, positionId: selectedPositionId });
      setSessions(res.data?.content || []);
    } catch {
      // silent — sidebar is non-critical
    }
  }, [selectedPositionId]);

  useEffect(() => { loadSessions(); }, [loadSessions]);

  // ── Scroll to bottom on new messages (suppressed during prepend) ──
  useEffect(() => {
    if (suppressScrollRef.current) {
      suppressScrollRef.current = false;
      return;
    }
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ── loadOlderMessages — defined before the IntersectionObserver effect ──
  const loadOlderMessages = useCallback(async () => {
    if (!currentSessionId || !hasOlderMessages || loadingOlder || firstMessageId === null) return;

    setLoadingOlder(true);
    const container = messagesContainerRef.current;
    const prevScrollHeight = container?.scrollHeight || 0;
    suppressScrollRef.current = true;

    try {
      const res = await chatbotApi.getSessionHistory(currentSessionId, {
        limit: CHAT_PAGE_SIZE,
        beforeId: firstMessageId,
      });
      const older = res.data || [];
      if (older.length === 0) {
        setHasOlderMessages(false);
        return;
      }
      const feMsgs = older.map(m => ({ id: m.id, role: mapRole(m.role), content: m.content }));
      setMessages(prev => [...feMsgs, ...prev]);
      setFirstMessageId(older[0].id);
      setHasOlderMessages(older.length === CHAT_PAGE_SIZE);

      // Maintain scroll position after DOM update
      requestAnimationFrame(() => {
        if (container) container.scrollTop = container.scrollHeight - prevScrollHeight;
      });
    } catch {
      suppressScrollRef.current = false;
    } finally {
      setLoadingOlder(false);
    }
  }, [currentSessionId, hasOlderMessages, loadingOlder, firstMessageId]);

  // ── IntersectionObserver — infinite scroll up ──
  useEffect(() => {
    if (!topSentinelRef.current || !hasOlderMessages || loadingOlder) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) loadOlderMessages(); },
      { root: messagesContainerRef.current, threshold: 0.1 }
    );
    observer.observe(topSentinelRef.current);
    return () => observer.disconnect();
  }, [hasOlderMessages, loadingOlder, loadOlderMessages]);

  // ── Select a session from sidebar ──
  const handleSelectSession = useCallback(async (session) => {
    if (currentSessionId === session.sessionId) return;

    setCurrentSessionId(session.sessionId);
    setSelectedPositionId(session.positionId || null);
    setMode(session.mode === 'INTERNAL' ? 'Internal' : 'External');
    navigate(session.positionId ? `/hr/chatbot/${session.positionId}` : '/hr/chatbot');

    setLoadingHistory(true);
    setMessages([]);
    setFirstMessageId(null);
    setHasOlderMessages(false);

    try {
      const res = await chatbotApi.getSessionHistory(session.sessionId, { limit: CHAT_PAGE_SIZE });
      const history = res.data || [];
      const feMsgs = history.map(m => ({ id: m.id, role: mapRole(m.role), content: m.content }));
      setMessages(feMsgs);
      if (history.length > 0) {
        setFirstMessageId(history[0].id);
        setHasOlderMessages(history.length === CHAT_PAGE_SIZE);
      }
    } catch {
      antMessage.error('Failed to load chat history');
    } finally {
      setLoadingHistory(false);
    }
  }, [currentSessionId, navigate]);

  // ── Position dropdown change ──
  const handlePositionChange = (val) => {
    setSelectedPositionId(val || null);
    navigate(val ? `/hr/chatbot/${val}` : '/hr/chatbot');
  };

  // ── New Chat ──
  const handleNewChatClick = () => {
    if (!selectedPositionId) {
      antMessage.warning('Please select a position first');
      return;
    }
    setModalMode('Internal');
    setShowModeModal(true);
  };

  const handleCreateSession = async () => {
    setCreatingSession(true);
    try {
      const beMode = modalMode === 'Internal' ? 'INTERNAL' : 'EXTERNAL';
      const res = await chatbotApi.createHRSession(user.id, selectedPositionId, beMode);
      setCurrentSessionId(res.session_id);
      setMode(modalMode);
      setMessages([{ role: 'assistant', content: 'Hello HR! I am your AI Assistant. How can I help you today?' }]);
      setFirstMessageId(null);
      setHasOlderMessages(false);
      setShowModeModal(false);
      await loadSessions();
    } catch {
      antMessage.error('Failed to create session');
    } finally {
      setCreatingSession(false);
    }
  };

  // ── Send message (SSE streaming) ──
  const handleSend = async () => {
    const content = inputValue.trim();
    if (!content) return;
    if (!currentSessionId) {
      antMessage.warning('Please start a new chat session first');
      return;
    }

    setMessages(prev => [...prev, { role: 'user', content }]);
    setInputValue('');
    streamBufferRef.current = '';
    setStreamingContent('');
    setStatusMessage('');
    setIsLoading(true);

    try {
      const beMode = mode === 'Internal' ? 'INTERNAL' : 'EXTERNAL';
      await chatbotApi.streamHRMessage(
        currentSessionId, content, user.id, selectedPositionId, beMode,
        {
          onStatus: (status) => setStatusMessage(status),
          onToken: (token) => {
            streamBufferRef.current += token;
            setStreamingContent(streamBufferRef.current);
          },
          onDone: ({ fallback_answer }) => {
            const finalContent = fallback_answer || streamBufferRef.current || '';
            setMessages(prev => [...prev, { role: 'assistant', content: finalContent }]);
            streamBufferRef.current = '';
            setStreamingContent('');
            setStatusMessage('');
          },
        }
      );
    } catch {
      antMessage.error('Failed to send message');
      setMessages(prev => prev.slice(0, -1));
      streamBufferRef.current = '';
      setStreamingContent('');
      setStatusMessage('');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── Derived ──
  const selectedPosition = positionMap[selectedPositionId];

  return (
    <>
      <Layout style={{ height: 'calc(100vh - 112px)', background: '#fff', borderRadius: '8px', overflow: 'hidden', border: '1px solid #E2E8F0', boxShadow: '0 2px 12px rgba(0,0,0,0.06)' }}>

        {/* ── Left Sidebar ── */}
        <Sider
          width={260}
          collapsedWidth={44}
          collapsed={sidebarCollapsed}
          theme="light"
          style={{ borderRight: '1px solid #E2E8F0', overflow: 'hidden', transition: 'width 0.2s', background: '#F8FAFC' }}
        >
          {/* Sidebar header: toggle + New Chat */}
          <div style={{
            padding: '12px 8px',
            borderBottom: '1px solid #E2E8F0',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            overflow: 'hidden',
          }}>
            {!sidebarCollapsed && (
              <Button type="primary" icon={<PlusOutlined />} block onClick={handleNewChatClick}>
                New Chat
              </Button>
            )}
            <Button
              type="text"
              icon={sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              style={{ flexShrink: 0 }}
            />
          </div>

          {/* Session list — hidden when collapsed */}
          {!sidebarCollapsed && (
            <div style={{ overflowY: 'auto', height: 'calc(100% - 57px)' }}>
              {sessions.length === 0 ? (
                <div style={{ padding: '16px', color: '#8c8c8c', fontSize: 13 }}>
                  {selectedPositionId ? 'No sessions yet.' : 'Select a position first.'}
                </div>
              ) : (
                <Menu
                  mode="inline"
                  selectedKeys={currentSessionId ? [currentSessionId] : []}
                  style={{ borderRight: 'none' }}
                  items={sessions.map(s => ({
                    key: s.sessionId,
                    label: (
                      <Tooltip title={getSessionTitle(s)} placement="right">
                        <Text ellipsis style={{ fontSize: 13, maxWidth: 188, display: 'block' }}>
                          {getSessionTitle(s)}
                        </Text>
                      </Tooltip>
                    ),
                    onClick: () => handleSelectSession(s),
                  }))}
                />
              )}
            </div>
          )}
        </Sider>

        {/* ── Right Content ── */}
        <Content style={{ display: 'flex', flexDirection: 'column' }}>

          {/* Header */}
          <div style={{ padding: '16px 24px', borderBottom: '1px solid #E2E8F0', background: '#fff', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
            <Space size={12} wrap>
              <Title level={5} style={{ margin: 0 }}>AI HR Assistant</Title>
              <Select
                allowClear
                placeholder="Select Position"
                style={{ width: 260 }}
                value={selectedPositionId}
                onChange={handlePositionChange}
                options={positions.map(p => ({ value: p.id, label: `${p.seniority} ${p.title}` }))}
              />
            </Space>
            <Space>
              {currentSessionId && (
                <Tag
                  color={mode === 'Internal' ? 'geekblue' : 'orange'}
                  style={{ fontWeight: 600, fontSize: 13, padding: '3px 10px' }}
                >
                  {mode}
                </Tag>
              )}
              <Button icon={<ReloadOutlined />} onClick={loadSessions}>
                Refresh
              </Button>
            </Space>
          </div>

          {/* Messages */}
          <div
            ref={messagesContainerRef}
            style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}
          >
            {/* Top sentinel — triggers infinite scroll up */}
            <div ref={topSentinelRef} style={{ height: 1, flexShrink: 0 }} />

            {loadingOlder && (
              <div style={{ textAlign: 'center', padding: '4px 0' }}>
                <Spin size="small" />
              </div>
            )}

            {loadingHistory ? (
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Spin size="large" />
              </div>
            ) : messages.length === 0 ? (
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ textAlign: 'center' }}>
                  <RobotOutlined style={{ fontSize: 56, marginBottom: 16, color: '#C7D2FE' }} />
                  <div style={{ color: '#94A3B8', maxWidth: 280, lineHeight: 1.6 }}>
                    Select a position and click{' '}
                    <strong style={{ color: '#4F46E5' }}>New Chat</strong>
                    , or resume a session from the sidebar.
                  </div>
                </div>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div
                  key={msg.id ?? idx}
                  className="chat-message"
                  style={{ display: 'flex', gap: '12px', flexDirection: msg.role === 'user' ? 'row-reverse' : 'row' }}
                >
                  <Avatar
                    icon={msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
                    style={{ backgroundColor: msg.role === 'user' ? '#4F46E5' : '#0F172A', flexShrink: 0 }}
                  />
                  <div style={{
                    maxWidth: '70%',
                    padding: '12px 16px',
                    borderRadius: msg.role === 'user' ? '18px 4px 18px 18px' : '4px 18px 18px 18px',
                    background: msg.role === 'user' ? '#4F46E5' : '#fff',
                    border: msg.role === 'user' ? 'none' : '1px solid #E2E8F0',
                    boxShadow: msg.role === 'user' ? '0 2px 8px rgba(79,70,229,0.25)' : '0 1px 4px rgba(0,0,0,0.06)',
                    color: msg.role === 'user' ? '#fff' : '#1E293B',
                  }}>
                    <ChatMarkdown>{msg.content}</ChatMarkdown>
                  </div>
                </div>
              ))
            )}

            {isLoading && (
              <div className="chat-message" style={{ display: 'flex', gap: '12px' }}>
                <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#0F172A', flexShrink: 0 }} />
                <div style={{
                  maxWidth: '70%',
                  padding: '12px 16px',
                  borderRadius: '4px 18px 18px 18px',
                  background: '#fff',
                  border: '1px solid #E2E8F0',
                  boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
                }}>
                  {streamingContent
                    ? <ChatMarkdown>{streamingContent}</ChatMarkdown>
                    : <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div className="typing-indicator">
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                        <div className="typing-dot" />
                      </div>
                      {statusMessage && (
                        <span style={{ color: '#94A3B8', fontSize: 13 }}>{statusMessage}</span>
                      )}
                    </div>
                  }
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div style={{ padding: '16px 24px', borderTop: '1px solid #E2E8F0', background: '#F8FAFC' }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
              <Input.TextArea
                autoSize={{ minRows: 2, maxRows: 6 }}
                placeholder={
                  currentSessionId
                    ? 'Enter your prompt... | Shift+Enter for new line'
                    : 'Start a new chat session to begin'
                }
                value={inputValue}
                onChange={e => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={!currentSessionId || isLoading}
                style={{ flex: 1 }}
              />
              <Button
                size="large"
                type="primary"
                icon={<SendOutlined />}
                onClick={handleSend}
                loading={isLoading}
                disabled={!currentSessionId}
                style={{ height: 'auto', padding: '10px 24px' }}
              >
                Send
              </Button>
            </div>
          </div>
        </Content>
      </Layout>

      {/* ── Mode Selection Modal ── */}
      <Modal
        title="Start New Chat"
        open={showModeModal}
        onOk={handleCreateSession}
        onCancel={() => setShowModeModal(false)}
        okText="Start Chat"
        confirmLoading={creatingSession}
        width={420}
      >
        <div style={{ padding: '16px 0' }}>
          {selectedPosition && (
            <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
              Position: <strong>{selectedPosition.seniority} {selectedPosition.title}</strong>
            </Text>
          )}
          <Segmented

            options={['Internal', 'External']}
            value={modalMode}
            onChange={setModalMode}
          />
          <Text type="secondary" style={{ marginTop: 12, display: 'block', fontSize: 13 }}>
            {modalMode === 'Internal'
              ? 'Search and analyze CVs uploaded by HR for this position.'
              : 'Search and analyze CVs submitted by Candidates for this position.'}
          </Text>
        </div>
      </Modal>
    </>
  );
};

export default HRChatbotPage;
