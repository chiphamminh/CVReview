import { useState, useRef, useEffect, useCallback } from 'react';
import { Drawer, Input, Button, Typography, Avatar, Result, Spin } from 'antd';
import { SendOutlined, RobotOutlined, UserOutlined, LoginOutlined, UploadOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import ChatMarkdown from '@/components/chatbot/ChatMarkdown';
import useAuthStore from '@/store/authStore';
import LearningPathCard from '@/components/candidate/LearningPathCard';
import { candidateApi } from '@/api/candidate.api';
import { chatbotApi } from '@/api/chatbot.api';

const { Title } = Typography;

const CHAT_PAGE_SIZE = 10;

const WELCOME_MSG = {
  role: 'assistant',
  content: 'Hello! I am your AI Recruitment Assistant. You can ask me about our company, open roles, or let me help you apply for a position based on your Master CV.',
};

const mapRole = (role) => (role === 'USER' ? 'user' : 'assistant');

const CandidateChatbotDrawer = ({ open, onClose }) => {
  const navigate = useNavigate();
  const { user, hasMasterCV, setHasMasterCV, token } = useAuthStore();

  const [cvId, setCvId] = useState(null);
  const [sessionId, setSessionId] = useState(null);

  const [checking, setChecking] = useState(false);
  const [initializingSession, setInitializingSession] = useState(false);

  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const [hasOlderMessages, setHasOlderMessages] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [firstMessageId, setFirstMessageId] = useState(null);

  const messagesContainerRef = useRef(null);
  const topSentinelRef = useRef(null);
  const suppressScrollRef = useRef(false);
  const messagesEndRef = useRef(null);

  // ── CV check: resolve hasMasterCV + cvId whenever drawer opens ──
  useEffect(() => {
    // Skip if no user, no valid token, or cvId already known
    if (!open || !user || !token || cvId !== null) return;

    let cancelled = false;
    if (hasMasterCV === null) setChecking(true);

    candidateApi.getMyCV()
      .then(res => {
        if (!cancelled) {
          setCvId(res.data?.cvId || null);
          if (hasMasterCV === null) setHasMasterCV(true);
        }
      })
      .catch(err => {
        if (!cancelled && hasMasterCV === null) {
          setHasMasterCV(err.response?.status === 404 ? false : null);
        }
      })
      // Always reset loading — even if cancelled (cleanup ran), spinner must not stay forever
      .finally(() => { setChecking(false); });

    return () => { cancelled = true; };
  }, [open, user, token, cvId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Session init: restore latest session or create new ──
  useEffect(() => {
    // token check prevents 401 from expired persisted auth state
    if (!open || !user || !token || hasMasterCV !== true || sessionId !== null) return;

    let cancelled = false;
    setInitializingSession(true);

    const init = async () => {
      try {
        const sessionsRes = await chatbotApi.getSessions({ page: 0, size: 5 });
        const allSessions = sessionsRes.data?.content || [];
        const candidateSessions = allSessions.filter(s => s.chatbotType === 'CANDIDATE');

        if (candidateSessions.length > 0 && !cancelled) {
          const existing = candidateSessions[0];
          // Note: setSessionId is called AFTER the await below.
          // Calling it before would trigger effect cleanup → cancelled=true → setMessages never runs.

          const histRes = await chatbotApi.getSessionHistory(existing.sessionId, { limit: CHAT_PAGE_SIZE });
          const history = histRes.data || [];

          if (!cancelled) {
            setSessionId(existing.sessionId);  // Batched with setMessages below — safe after await
            if (history.length > 0) {
              const feMsgs = history.map(m => ({ id: m.id, role: mapRole(m.role), content: m.content }));
              setMessages(feMsgs);
              setFirstMessageId(history[0].id);
              setHasOlderMessages(history.length === CHAT_PAGE_SIZE);
            } else {
              setMessages([WELCOME_MSG]);
            }
          }
        } else if (!cancelled) {
          const res = await chatbotApi.createCandidateSession(user.id);
          if (!cancelled) {
            setSessionId(res.session_id);
            setMessages([WELCOME_MSG]);
          }
        }
      } catch (err) {
        // 401: token expired — axiosClient will redirect to login
        // Other errors: show welcome so UI is not stuck
        if (!cancelled && err?.response?.status !== 401) {
          setMessages([WELCOME_MSG]);
        }
      } finally {
        // Always reset — even if cancelled, spinner must not stay forever
        setInitializingSession(false);
      }
    };

    init();
    return () => { cancelled = true; };
  }, [open, user, token, hasMasterCV, sessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Scroll to bottom on new messages (suppressed during prepend) ──
  useEffect(() => {
    if (!open) return;
    if (suppressScrollRef.current) {
      suppressScrollRef.current = false;
      return;
    }
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, open]);

  // ── loadOlderMessages ──
  const loadOlderMessages = useCallback(async () => {
    if (!sessionId || !hasOlderMessages || loadingOlder || firstMessageId === null) return;

    setLoadingOlder(true);
    const container = messagesContainerRef.current;
    const prevScrollHeight = container?.scrollHeight || 0;
    suppressScrollRef.current = true;

    try {
      const res = await chatbotApi.getSessionHistory(sessionId, {
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

      requestAnimationFrame(() => {
        if (container) container.scrollTop = container.scrollHeight - prevScrollHeight;
      });
    } catch {
      suppressScrollRef.current = false;
    } finally {
      setLoadingOlder(false);
    }
  }, [sessionId, hasOlderMessages, loadingOlder, firstMessageId]);

  // ── IntersectionObserver — infinite scroll up ──
  useEffect(() => {
    if (!open || !topSentinelRef.current || !hasOlderMessages || loadingOlder) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) loadOlderMessages(); },
      { root: messagesContainerRef.current, threshold: 0.1 }
    );
    observer.observe(topSentinelRef.current);
    return () => observer.disconnect();
  }, [open, hasOlderMessages, loadingOlder, loadOlderMessages]);

  // ── Send message ──
  const handleSend = async (text = inputValue) => {
    const content = text.trim();
    if (!content || !sessionId) return;

    setMessages(prev => [...prev, { role: 'user', content }]);
    setInputValue('');
    setIsLoading(true);

    try {
      const res = await chatbotApi.sendCandidateMessage(sessionId, content, user.id, cvId);
      setMessages(prev => [...prev, { role: 'assistant', content: res.answer }]);
    } catch {
      setMessages(prev => prev.slice(0, -1));
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

  // ── Render ──
  const renderContent = () => {
    if (!user) {
      return (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }}>
          <Result
            icon={<RobotOutlined style={{ color: '#1677ff' }} />}
            title="Login Required"
            subTitle="Please login or register to use the AI Recruitment Assistant."
            extra={[
              <Button type="primary" icon={<LoginOutlined />} key="login" onClick={() => { onClose(); navigate('/login'); }}>
                Login / Register
              </Button>,
            ]}
          />
        </div>
      );
    }

    if (checking || initializingSession) {
      return (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Spin size="large" />
        </div>
      );
    }

    if (hasMasterCV === false) {
      return (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }}>
          <Result
            icon={<RobotOutlined style={{ color: '#faad14' }} />}
            title="Upload Your CV First"
            subTitle="The AI Assistant needs your Master CV to evaluate your fit and help you apply for positions."
            extra={[
              <Button type="primary" icon={<UploadOutlined />} key="upload" onClick={() => { onClose(); navigate('/candidate/cv'); }}>
                Upload CV
              </Button>,
            ]}
          />
        </div>
      );
    }

    return (
      <>
        <div
          ref={messagesContainerRef}
          style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px', background: '#fff' }}
        >
          {/* Top sentinel */}
          <div ref={topSentinelRef} style={{ height: 1, flexShrink: 0 }} />

          {loadingOlder && (
            <div style={{ textAlign: 'center', padding: '4px 0' }}>
              <Spin size="small" />
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={msg.id ?? idx} style={{ display: 'flex', gap: '12px', flexDirection: msg.role === 'user' ? 'row-reverse' : 'row' }}>
              <Avatar
                icon={msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
                style={{ backgroundColor: msg.role === 'user' ? '#1677ff' : '#52c41a', flexShrink: 0 }}
              />
              <div style={{ maxWidth: '80%', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{
                  padding: '10px 14px',
                  borderRadius: '8px',
                  backgroundColor: msg.role === 'user' ? '#e6f4ff' : '#f5f5f5',
                  border: msg.role === 'user' ? '1px solid #91caff' : '1px solid #d9d9d9',
                }}>
                  <ChatMarkdown>{msg.content}</ChatMarkdown>
                </div>
                {msg.isRejection && (
                  <LearningPathCard
                    score={msg.score}
                    missingSkills={msg.missingSkills}
                    learningPathText={msg.learningPathText}
                  />
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div style={{ display: 'flex', gap: '12px' }}>
              <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#52c41a' }} />
              <div style={{ padding: '10px 14px', color: '#8c8c8c' }}>AI is thinking...</div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div style={{ padding: '16px', borderTop: '1px solid #f0f0f0', background: '#fafafa' }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
            <Input.TextArea
              autoSize={{ minRows: 2, maxRows: 6 }}
              placeholder="Type your message..."
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              style={{ flex: 1 }}
            />
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={() => handleSend()}
              loading={isLoading}
              style={{ height: 'auto', padding: '10px 16px' }}
            />
          </div>
        </div>
      </>
    );
  };

  return (
    <Drawer
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <RobotOutlined style={{ color: '#52c41a' }} />
          <Title level={5} style={{ margin: 0 }}>AI Assistant</Title>
        </div>
      }
      placement="right"
      width={500}
      onClose={onClose}
      open={open}
      styles={{ body: { display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' } }}
    >
      {renderContent()}
    </Drawer>
  );
};

export default CandidateChatbotDrawer;
