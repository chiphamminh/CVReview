import { useState, useRef, useEffect } from 'react';
import { Drawer, Input, Button, Typography, Avatar, Result, Spin } from 'antd';
import { SendOutlined, RobotOutlined, UserOutlined, LoginOutlined, UploadOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import useAuthStore from '@/store/authStore';
import LearningPathCard from '@/components/candidate/LearningPathCard';
import { candidateApi } from '@/api/candidate.api';

const { Title } = Typography;

const CandidateChatbotDrawer = ({ open, onClose }) => {
  const navigate = useNavigate();
  const { user, hasMasterCV, setHasMasterCV } = useAuthStore();

  const [checking, setChecking] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am your AI Recruitment Assistant. You can ask me about our company, open roles, or let me help you apply for a position based on your Master CV.' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // When drawer opens and user is logged in but CV status unknown → resolve it once
  useEffect(() => {
    if (!open || !user || hasMasterCV !== null) return;

    let cancelled = false;
    setChecking(true);
    candidateApi.getMyCV()
      .then(() => { if (!cancelled) setHasMasterCV(true); })
      .catch((err) => {
        if (!cancelled) setHasMasterCV(err.response?.status === 404 ? false : null);
      })
      .finally(() => { if (!cancelled) setChecking(false); });

    return () => { cancelled = true; };
  }, [open, user, hasMasterCV, setHasMasterCV]);

  useEffect(() => {
    if (open) messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, open]);

  const handleSend = (text = inputValue) => {
    const content = text.trim();
    if (!content) return;

    const userMsg = { role: 'user', content };
    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    setTimeout(() => {
      let aiResponseMsg = {};
      if (content.toLowerCase().includes('apply')) {
        const mockScore = 65;
        aiResponseMsg = {
          role: 'assistant',
          content: 'I have analyzed your Master CV against this position.',
          isRejection: true,
          score: mockScore,
          missingSkills: ['Microservices', 'GraphQL', 'AWS'],
          learningPathText: 'Your current experience is slightly below our expectations for this role. We recommend building 1-2 projects involving Microservices architecture and getting familiar with AWS deployments.\n- AWS Certified Developer Associate Course\n- Spring Boot Microservices Tutorial'
        };
      } else {
        aiResponseMsg = {
          role: 'assistant',
          content: `I received your query: **${content}**. How else can I assist you?`
        };
      }
      setMessages(prev => [...prev, aiResponseMsg]);
      setIsLoading(false);
    }, 1500);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const renderContent = () => {
    // Not logged in
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
              </Button>
            ]}
          />
        </div>
      );
    }

    // Checking CV status
    if (checking) {
      return (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Spin size="large" />
        </div>
      );
    }

    // No Master CV
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
              </Button>
            ]}
          />
        </div>
      );
    }

    // Normal chat UI
    return (
      <>
        <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px', background: '#fff' }}>
          {messages.map((msg, idx) => (
            <div key={idx} style={{ display: 'flex', gap: '12px', flexDirection: msg.role === 'user' ? 'row-reverse' : 'row' }}>
              <Avatar
                icon={msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
                style={{ backgroundColor: msg.role === 'user' ? '#1677ff' : '#52c41a' }}
              />
              <div style={{ maxWidth: '80%', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{
                  padding: '10px 14px',
                  borderRadius: '8px',
                  backgroundColor: msg.role === 'user' ? '#e6f4ff' : '#f5f5f5',
                  border: msg.role === 'user' ? '1px solid #91caff' : '1px solid #d9d9d9',
                }}>
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
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
              style={{ flex: 1 }}
            />
            <Button type="primary" icon={<SendOutlined />} onClick={() => handleSend()} loading={isLoading} style={{ height: 'auto', padding: '10px 16px' }} />
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
      styles={{
        body: { display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' }
      }}
    >
      {renderContent()}
    </Drawer>
  );
};

export default CandidateChatbotDrawer;
