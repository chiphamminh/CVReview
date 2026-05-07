import React, { useState, useRef, useEffect } from 'react';
import { Drawer, Input, Button, Typography, Avatar, Select } from 'antd';
import { SendOutlined, RobotOutlined, UserOutlined } from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import LearningPathCard from '@/components/candidate/LearningPathCard';

const { Title } = Typography;

const mockSessions = [
  { value: '1', label: 'Applying for Java Backend' },
  { value: '2', label: 'Career Advice' },
  { value: 'new', label: '+ Start New Chat' },
];

const CandidateChatbotDrawer = ({ open, onClose }) => {
  const [selectedSession, setSelectedSession] = useState('new');
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am your AI Recruitment Assistant. You can ask me about our company, open roles, or let me help you apply for a position based on your Master CV.' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (open) {
      scrollToBottom();
    }
  }, [messages, open]);

  const handleSend = (text = inputValue) => {
    const content = text.trim();
    if (!content) return;

    const userMsg = { role: 'user', content };
    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    // Mock AI response logic
    setTimeout(() => {
      let aiResponseMsg = {};
      
      // Giả lập case nộp đơn bị fail do score < 70 (Guardrail)
      if (content.toLowerCase().includes('apply')) {
        const mockScore = 65; 
        aiResponseMsg = {
          role: 'assistant',
          content: 'I have analyzed your Master CV against this position.',
          isRejection: true,
          score: mockScore,
          missingSkills: ['Microservices', 'GraphQL', 'AWS'],
          learningPathText: 'Your current experience is slightly below our expectations for this role. We recommend building 1-2 projects involving Microservices architecture and getting familiar with AWS deployments. Here are some useful resources: \n- AWS Certified Developer Associate Course\n- Spring Boot Microservices Tutorial'
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

  const handleSessionChange = (val) => {
    setSelectedSession(val);
    if (val === 'new') {
      setMessages([
        { role: 'assistant', content: 'Hello! I am your AI Recruitment Assistant. You can ask me about our company, open roles, or let me help you apply for a position based on your Master CV.' }
      ]);
    } else {
      // Mock loading history
      setMessages([
        { role: 'assistant', content: `Loaded history for session ${val}.` }
      ]);
    }
  };

  return (
    <Drawer
      title={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Title level={5} style={{ margin: 0 }}>AI Assistant</Title>
          <Select 
            value={selectedSession}
            onChange={handleSessionChange}
            options={mockSessions}
            style={{ width: 180 }}
            size="small"
          />
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
      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px', background: '#fff' }}>
        {messages.map((msg, idx) => (
          <div key={idx} style={{ 
            display: 'flex', 
            gap: '12px', 
            flexDirection: msg.role === 'user' ? 'row-reverse' : 'row' 
          }}>
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
              {/* Render Guardrail UI if applicable */}
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

      {/* Input */}
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
    </Drawer>
  );
};

export default CandidateChatbotDrawer;
