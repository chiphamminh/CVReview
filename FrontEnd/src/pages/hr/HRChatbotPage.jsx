import React, { useState, useRef, useEffect } from 'react';
import { Layout, Menu, Input, Button, Typography, Space, Segmented, Avatar, Select } from 'antd';
import { SendOutlined, RobotOutlined, UserOutlined, PlusOutlined } from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { fetchActivePositions } from '@/api/mockData';

const { Sider, Content } = Layout;
const { Title, Text } = Typography;

// Mock Chat History
const mockSessions = [
  { id: '1', title: 'Analyze Java Backend JD' },
  { id: '2', title: 'Filter Senior React Candidates' },
  { id: '3', title: 'Generate Interview Questions' },
];

const HRChatbotPage = () => {
  const { positionId } = useParams();
  const navigate = useNavigate();
  const [mode, setMode] = useState('Internal'); // Internal | External
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello HR! I am your AI Assistant. How can I help you today?' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const { data: activePositions } = useQuery({
    queryKey: ['activePositions'],
    queryFn: fetchActivePositions,
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = () => {
    if (!inputValue.trim()) return;

    const userMsg = { role: 'user', content: inputValue };
    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    // Mock AI response
    setTimeout(() => {
      const aiResponse = `Received your request: **${userMsg.content}** in **${mode}** mode.\n\nHere are the mock analysis results from the system...`;
      setMessages(prev => [...prev, { role: 'assistant', content: aiResponse }]);
      setIsLoading(false);
    }, 1000);
  };

  const handleKeyDown = (e) => {
    // Submit on Enter, allow Shift+Enter for new line
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handlePositionChange = (val) => {
    if (val) {
      navigate(`/hr/chatbot/${val}`);
    } else {
      navigate(`/hr/chatbot`);
    }
  };

  return (
    <Layout style={{ height: 'calc(100vh - 112px)', background: '#fff', borderRadius: '8px', overflow: 'hidden' }}>
      {/* Left Sidebar: Chat History */}
      <Sider width={250} theme="light" style={{ borderRight: '1px solid #f0f0f0', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '16px', borderBottom: '1px solid #f0f0f0' }}>
          <Button type="primary" icon={<PlusOutlined />} block>New Chat</Button>
        </div>
        <Menu
          mode="inline"
          defaultSelectedKeys={['1']}
          style={{ borderRight: 'none', flex: 1, overflowY: 'auto' }}
          items={mockSessions.map(session => ({
            key: session.id,
            label: session.title,
          }))}
        />
      </Sider>

      {/* Right Content: Chat Area */}
      <Content style={{ display: 'flex', flexDirection: 'column' }}>
        {/* Chat Header */}
        <div style={{ padding: '16px 24px', borderBottom: '1px solid #f0f0f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16 }}>
          <Space>
            <Title level={5} style={{ margin: 0 }}>AI HR Assistant</Title>
            <Select 
              allowClear
              placeholder="Select Active Position"
              style={{ width: 250, marginLeft: 16 }}
              value={positionId ? parseInt(positionId) : null}
              onChange={handlePositionChange}
              options={activePositions?.map(p => ({ value: p.id, label: p.name })) || []}
            />
          </Space>
          <Segmented 
            options={[
              { label: 'Internal (HR Mode)', value: 'Internal' },
              { label: 'External (Candidate Mode)', value: 'External' }
            ]} 
            value={mode}
            onChange={setMode}
          />
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
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
              <div style={{
                maxWidth: '70%',
                padding: '12px 16px',
                borderRadius: '8px',
                backgroundColor: msg.role === 'user' ? '#e6f4ff' : '#f5f5f5',
                border: msg.role === 'user' ? '1px solid #91caff' : '1px solid #d9d9d9',
              }}>
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
            </div>
          ))}
          {isLoading && (
            <div style={{ display: 'flex', gap: '12px' }}>
              <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#52c41a' }} />
              <div style={{ padding: '12px', color: '#8c8c8c' }}>AI is thinking...</div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div style={{ padding: '16px 24px', borderTop: '1px solid #f0f0f0', background: '#fafafa' }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
            <Input.TextArea
              autoSize={{ minRows: 2, maxRows: 6 }}
              placeholder="Enter your prompt for AI (e.g. Filter top 5 candidates for this position...) | Shift + Enter for new line"
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              style={{ flex: 1 }}
            />
            <Button size="large" type="primary" icon={<SendOutlined />} onClick={handleSend} loading={isLoading} style={{ height: 'auto', padding: '10px 24px' }}>
              Send
            </Button>
          </div>
        </div>
      </Content>
    </Layout>
  );
};

export default HRChatbotPage;
